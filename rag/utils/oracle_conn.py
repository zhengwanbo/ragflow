import logging
import json
import os
import re
import oracledb
import copy
from typing import Any, List, Dict
from rag import settings
from rag.settings import TAG_FLD, PAGERANK_FLD
from rag.utils import singleton
from rag.nlp import is_english, rag_tokenizer
from api.utils.file_utils import get_project_base_directory
from rag.utils.doc_store_conn import (
    DocStoreConnection,
    MatchExpr,
    MatchTextExpr,
    MatchDenseExpr,
    FusionExpr,
    OrderByExpr,
)

logger = logging.getLogger('ragflow.oracle_conn')
#logger.setLevel(logging.DEBUG)

def equivalent_condition_to_str(condition: dict, table_instance=None) -> str:
    """
    将条件字典转换为SQL查询条件字符串
    :param condition: 条件字典
    :param table_instance: 表实例（Oracle中暂不使用）
    :return: SQL条件字符串
    """
    where_clauses = []
    
    # 处理id条件
    if "id" in condition:
        chunk_ids = condition["id"]
        if not isinstance(chunk_ids, list):
            chunk_ids = [chunk_ids]
        # 处理字符串ID
        if isinstance(chunk_ids[0], str):
            values = ", ".join([f"'{x}'" for x in chunk_ids])
        else:
            values = ", ".join([str(x) for x in chunk_ids])

        return f"id IN ({values})"
    
    for k, v in condition.items():
        if k == "id":
            continue

        # 处理exists条件
        if k == "exists":
            where_clauses.append(f"{v} IS NOT NULL")
            continue
            
        # 处理must_not exists条件
        if k == "must_not":
            if isinstance(v, dict) and "exists" in v:
                where_clauses.append(f"{v['exists']} IS NULL")
            continue
            
        # 处理列表条件（terms）
        if isinstance(v, list):
            if not v:
                continue
            # 处理字符串列表
            if isinstance(v[0], str):
                values = ", ".join([f"'{x}'" for x in v])
            else:
                values = ", ".join([str(x) for x in v])
            where_clauses.append(f"{k} IN ({values})")
            continue
            
        # 处理字符串和数字条件
        if isinstance(v, str):
            where_clauses.append(f"{k} = '{v}'")
        elif isinstance(v, int) or isinstance(v, float):
            where_clauses.append(f"{k} = {v}")
        else:
            raise ValueError(f"Unsupported condition type: {type(v)} for field {k}")
    
    return " AND ".join(where_clauses) if where_clauses else "1=1"

@singleton
class OracleConnection(DocStoreConnection):
    def __init__(self):
        self.info = {}
        logger.info(f"Use Oracle 23ai {settings.ORACLE['host']} as the doc engine.")
        self.db_name = "freepdb1"
        self.conn_pool = None
        self._init_connection_pool()
        
    def _init_connection_pool(self):
        """初始化Oracle连接池"""
        try:
            self.conn_pool = oracledb.create_pool(
                user=settings.ORACLE["user"],
                password=settings.ORACLE["password"],
                dsn="{}:{}/{}".format(settings.ORACLE["host"], settings.ORACLE["port"], settings.ORACLE["db"]),
                min=1,
                max=10,
                increment=1
            )
            logger.info("Oracle connection pool created successfully")
        except Exception as e:
            logger.error(f"Failed to create Oracle connection pool: {str(e)}")
            raise

    def dbType(self) -> str:
        return "ORACLE"

    def health(self) -> dict:
        """检查数据库健康状态"""
        try:
            with self.conn_pool.acquire() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT 1 FROM DUAL")
                result = cursor.fetchone()
                return {
                    "status": "green" if result[0] == 1 else "red",
                    "message": "Oracle database is healthy" if result[0] == 1 else "Oracle database is not healthy"
                }
        except Exception as e:
            return {
                "status": "red",
                "message": str(e)
            }

    def createIdx(self, indexName: str, knowledgebaseId: str, vectorSize: int) -> bool:
        """创建表"""
        try:
            table_name = f"{indexName}_{knowledgebaseId}"
            logger.info(f"Created table: {table_name} ")

            with self.conn_pool.acquire() as conn:
                cursor = conn.cursor()
                
                # 读取oracle_mapping.json中的列定义
                fp_mapping = os.path.join(
                    get_project_base_directory(), "conf", "oracle_mapping.json"
                )
                if not os.path.exists(fp_mapping):
                    raise Exception(f"Mapping file not found at {fp_mapping}")
                schema = json.load(open(fp_mapping))
                
                # 添加向量列
                vector_name = f"q_{vectorSize}_vec"
                schema[vector_name] = {"type": f"VECTOR({vectorSize})"}
                
                # 构建列定义
                columns = []
                for col_name, col_info in schema.items():
                    col_type = col_info["type"]
                    if "default" in col_info:
                        default_value = col_info["default"]
                        if isinstance(default_value, str):
                            default_value = f"'{default_value}'"
                        columns.append(f"{col_name} {col_type} DEFAULT {default_value}")
                    else:
                        columns.append(f"{col_name} {col_type}")
                
                # 创建表
                create_sql = f"""
                CREATE TABLE {table_name} (
                    {", ".join(columns)}
                )
                """

                logger.info(f"Created table SQL: {create_sql} ")

                cursor.execute(create_sql)
                
                # 创建全文索引
                for field_name, field_info in schema.items():
                    if field_info["type"] == "CLOB" and "analyzer" in field_info:
                        index_sql = f"""
                        CREATE INDEX idx_{table_name}_{field_name} ON {table_name}({field_name})
                        INDEXTYPE IS CTXSYS.CONTEXT
                        PARAMETERS ('LEXER sys.my_chinese_vgram_lexer')
                        """

                        cursor.execute(index_sql)
                
                # Create vector index
                #cursor.execute(f"""
                #    CREATE VECTOR INDEX {table_name}_vec_idx 
                #    ON {table_name}(q_{vectorSize}_vec)
                #    ORGANIZATION INMEMORY NEIGHBOR GRAPH
                #""")
#
#                logger.info(f"Created table {table_name} with vector index")
                
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Failed to create table {table_name}: {str(e)}")
            return False

    def deleteIdx(self, indexName: str, knowledgebaseId: str):
        table_name = f"{indexName}_{knowledgebaseId}"
        with self.conn_pool.acquire() as conn:
            with conn.cursor() as cursor:
                cursor.execute(f"DROP TABLE {table_name} CASCADE CONSTRAINTS")
                logger.info(f"Dropped table {table_name}")

    def indexExist(self, indexName: str, knowledgebaseId: str) -> bool:
        table_name = f"{indexName}_{knowledgebaseId}"
        with self.conn_pool.acquire() as conn:
            with conn.cursor() as cursor:
                cursor.execute(f"""
                    SELECT COUNT(*) 
                    FROM user_tables 
                    WHERE table_name = UPPER('{table_name}')
                """)
                return cursor.fetchone()[0] > 0

    def insert(
            self, documents: List[Dict], indexName: str, knowledgebaseId: str = None
    ) -> List[str]:
        """插入文档"""
        try:
            table_name = f"{indexName}_{knowledgebaseId}" if knowledgebaseId else indexName
            
            with self.conn_pool.acquire() as conn:
                cursor = conn.cursor()
                
                # 检查表是否存在
                try:
                    # 尝试查询表结构
                    cursor.execute(f"SELECT * FROM {table_name} WHERE 1=0")
                except oracledb.DatabaseError as e:
                    # 表不存在时创建新表
                    error, = e.args
                    if error.code == 942:  # ORA-00942: table or view does not exist
                        # 推断向量维度
                        vector_size = 0
                        patt = re.compile(r"q_(?P<vector_size>\d+)_vec")
                        for k in documents[0].keys():
                            m = patt.match(k)
                            if m:
                                vector_size = int(m.group("vector_size"))
                                break
                        if vector_size == 0:
                            raise ValueError("Cannot infer vector size from documents")
                        
                        # 创建新表
                        self.createIdx(indexName, knowledgebaseId, vector_size)
                    else:
                        raise

                # 数据预处理
                docs = copy.deepcopy(documents)
                for d in docs:
                    assert "_id" not in d
                    assert "id" in d
                    for k, v in d.items():
                        if k in ["important_kwd", "question_kwd", "entities_kwd", "tag_kwd", "source_id"]:
                            assert isinstance(v, list)
                            d[k] = "###".join(v)
                        elif re.search(r"_feas$", k):
                            d[k] = json.dumps(v)
                        elif k == 'kb_id':
                            if isinstance(d[k], list):
                                d[k] = d[k][0]  # since d[k] is a list, but we need a str
                        elif k == "position_int":
                            assert isinstance(v, list)
                            arr = [num for row in v for num in row]
                            d[k] = "_".join(f"{num:08x}" for num in arr)
                        elif k in ["page_num_int", "top_int"]:
                            assert isinstance(v, list)
                            d[k] = "_".join(f"{num:08x}" for num in v)

                # 获取第一个文档的字段作为列名
                columns = list(docs[0].keys())
                placeholders = [f":{i+1}" for i in range(len(columns))]
                
                # 先删除已存在的记录
                ids = ["'{}'".format(d["id"]) for d in docs]
                str_ids = ", ".join(ids)
                delete_sql = f"DELETE FROM {table_name} WHERE id IN ({str_ids})"
                cursor.execute(delete_sql)

                #logger.debug(f"ORACLE insert DOCs: {docs}")
                
                # 插入新记录
                insert_sql = f"""
                INSERT INTO {table_name} ({", ".join(columns)})
                VALUES ({", ".join([f":{col}" for col in columns])})
                """

                # 准备批量插入数据
                data = []
                for doc in docs:
                    # 将文档转换为字典格式，键为列名
                    row_data = {}
                    for col in columns:
                        value = doc[col]
                        # 对向量字段进行特殊处理
                        if col.startswith('q_') and col.endswith('_vec') and isinstance(value, list):
                            row_data[col] = f'{json.dumps(value)}'  # 转换为JSON字符串并添加单引号
                        else:
                            row_data[col] = value
                    data.append(row_data)

                logger.debug(f"ORACLE inserted data  {insert_sql}, {data}.")
                # 执行批量插入
                cursor.executemany(insert_sql, data)

                conn.commit()
                logger.debug(f"ORACLE inserted into {table_name} {str_ids}.")

                return []
        except Exception as e:
            logger.error(f"Failed to insert into table {table_name}: {str(e)}")
            return []

    def get(self, chunkId: str, indexName: str, 
            knowledgebaseIds: list[str]) -> dict | None:
        assert isinstance(knowledgebaseIds, list)
        result = None
        
        with self.conn_pool.acquire() as conn:
            with conn.cursor() as cursor:
                for knowledgebaseId in knowledgebaseIds:
                    table_name = f"{indexName}_{knowledgebaseId}"
                    
                    # 检查表是否存在
                    cursor.execute(f"""
                        SELECT COUNT(*) 
                        FROM user_tables 
                        WHERE table_name = UPPER('{table_name}')
                    """)
                    if cursor.fetchone()[0] == 0:
                        logger.warning(
                            f"Table not found: {table_name}, this knowledge base isn't created in Oracle. Maybe it is created in other document engine.")
                        continue
                    
                    # 查询数据
                    cursor.execute(f"""
                        SELECT * FROM {table_name}
                        WHERE id = :id
                    """, id=chunkId)
                    
                    # 获取结果
                    row = cursor.fetchone()
                    if row:
                        # 将结果转换为字典
                        columns = [col[0] for col in cursor.description]
                        result = dict(zip(columns, row))
                        break
                        
        if result:
            # 处理特殊字段
            for field in ["important_kwd", "question_kwd", "entities_kwd", "tag_kwd", "source_id"]:
                if field in result and isinstance(result[field], str):
                    result[field] = result[field].split("###")
            
            # 处理position_int字段
            if "position_int" in result and isinstance(result["position_int"], str):
                hex_values = result["position_int"].split("_")
                result["position_int"] = [
                    [int(hex_values[i], 16) for i in range(j, j+5)]
                    for j in range(0, len(hex_values), 5)
                ]
            
            # 处理page_num_int和top_int字段
            for field in ["page_num_int", "top_int"]:
                if field in result and isinstance(result[field], str):
                    result[field] = [int(hex_val, 16) for hex_val in result[field].split("_")]
        
        #logger.debug(f"ORACLE get result: {result}")
        return result    
        
    def search(
        self, selectFields: list[str],
        highlightFields: list[str],
        condition: dict,
        matchExprs: list[MatchExpr],
        orderBy: OrderByExpr,
        offset: int,
        limit: int,
        indexNames: str | list[str],
        knowledgebaseIds: list[str],
        aggFields: list[str] = [],
        rank_feature: dict | None = None
    ) -> tuple[list[dict], int]:
        if isinstance(indexNames, str):
            indexNames = indexNames.split(",")
        assert isinstance(indexNames, list) and len(indexNames) > 0

        total_hits_count = 0
        results = []
        
        with self.conn_pool.acquire() as conn:
            with conn.cursor() as cursor:
                # 处理select字段
                select_fields = [f for f in selectFields if f != "_score"]
                if "id" not in select_fields:
                    select_fields.append("id")
                
                # 处理排序
                order_by_clause = ""
                if orderBy.fields:
                    order_by_clause = "ORDER BY " + ", ".join(
                        [f"{field} {'ASC' if sort == 0 else 'DESC'}" 
                         for field, sort in orderBy.fields]
                    )
                
                # 遍历所有表和知识库
                for indexName in indexNames:
                    for knowledgebaseId in knowledgebaseIds:
                        table_name = f"{indexName}_{knowledgebaseId}"
                        
                        # 检查表是否存在
                        cursor.execute(f"""
                            SELECT COUNT(*) 
                            FROM user_tables 
                            WHERE table_name = UPPER('{table_name}')
                        """)
                        if cursor.fetchone()[0] == 0:
                            continue
                            
                        # 构建基础查询
                        sql = f"SELECT {', '.join(select_fields)} FROM {table_name}"
                        params = {}
                        
                        # 处理匹配表达式
                        where_clauses = []
                        vector_similarity_weight = 0.5
                        for matchExpr in matchExprs:
                            if isinstance(matchExpr, MatchTextExpr):
                                # 处理字段权重
                                fields = []
                                for field in matchExpr.fields:
                                    if '^' in field:
                                        field_name, _ = field.split('^')
                                        fields.append(field_name)
                                    else:
                                        fields.append(field)
                                
                                # 构建字段合并表达式
                                fields_expr = " || ".join(fields)
                                
                                # 处理匹配文本
                                query_text = matchExpr.matching_text
                                # 去掉权重和模糊匹配符号
                                query_text = re.sub(r'\^[0-9.]+', '', query_text)  # 去掉权重
                                query_text = re.sub(r'~[0-9]+', '', query_text)  # 去掉模糊匹配
                                # 在括号之间添加OR
                                query_text = re.sub(r'\)\s*\(', ') OR (', query_text)
                                query_text = f"({query_text})"

                                # 构建CONTAINS条件
                                where_clauses.append(f"""
                                    CONTAINS(
                                        content_ltks,
                                        :text_query,
                                        1
                                    ) > 0
                                """)
                                
                                # 添加排序
                                order_by_clause = "ORDER BY SCORE(1) DESC"
                                
                                params["text_query"] = query_text
                                logger.info(f"Oracle search MatchTextExpr: {json.dumps(matchExpr.__dict__)}")
                                logger.info(f"ORACLE search MatchTextExpr Params: {params}")

                            elif isinstance(matchExpr, MatchDenseExpr):
                                # 向量检索
                                vector_column = matchExpr.vector_column_name
                                embedding_data = matchExpr.embedding_data
                                distance_type = matchExpr.distance_type
                                similarity_threshold = matchExpr.extra_options.get("similarity", 0.8)
                                
                                logger.info(f"Oracle Vector search similarity_threshold: {similarity_threshold}")
                                # 构建向量查询条件
                                where_clauses.append(f"""
                                    VECTOR_DISTANCE(
                                        {vector_column},
                                        :{vector_column}_query,
                                        {distance_type}
                                    ) < :{vector_column}_threshold
                                """)
                                
                                # 添加参数
                                params[f"{vector_column}_query"] = json.dumps(embedding_data)
                                params[f"{vector_column}_threshold"] = 1 - similarity_threshold
                                #logger.debug(f"Oracle search MatchDenseExpr: {json.dumps(matchExpr.__dict__)}")
                            elif isinstance(matchExpr, FusionExpr):
                                logger.debug(f"Oracle search FusionExpr: {json.dumps(matchExpr.__dict__)}")


                        # 添加条件过滤
                        if condition:
                            where_clauses.append(equivalent_condition_to_str(condition))
                            
                        # 组合where子句
                        if where_clauses:
                            sql += " WHERE " + " AND ".join(where_clauses)
                            
                        # 添加排序和分页
                        sql += f"""
                            {order_by_clause}
                            OFFSET {offset} ROWS FETCH NEXT {limit} ROWS ONLY
                        """
                        
                        logger.info(f"ORACLE search SQL: {sql}")

                        # 执行查询
                        cursor.execute(sql, params)
                        rows = cursor.fetchall()
                        total_hits_count += len(rows)
                        
                        # 转换结果格式
                        columns = [col[0].lower() for col in cursor.description]

                        # 转换结果格式
                        for row in rows:
                            # 处理CLOB字段
                            row_dict = {}
                            for idx, value in enumerate(row):
                                if isinstance(value, oracledb.LOB):  # 如果是LOB类型
                                    row_dict[columns[idx]] = value.read()  # 读取LOB内容
                                else:
                                    row_dict[columns[idx]] = value
                            results.append(row_dict)

        
        # 处理排序
        if matchExprs:
            # 如果有匹配表达式，按分数排序
            results.sort(key=lambda x: x.get("score", 0) + x.get(PAGERANK_FLD, 0), reverse=True)
        
        # 限制结果数量
        results = results[:limit]
        logger.info(f"ORACLE search final result: {total_hits_count}")
        return results, total_hits_count

    def update(
            self,
            indexName: str,
            knowledgebaseId: str,
            condition: Dict[str, Any],
            updates: Dict[str, Any]
    ) -> bool:
        """更新文档"""
        try:
            table_name = f"{indexName}_{knowledgebaseId}"
            with self.conn_pool.acquire() as conn:
                cursor = conn.cursor()
                
                # 构建SET子句
                set_clause = ", ".join([f"{k} = :{k}" for k in updates.keys()])
                
                # 构建WHERE子句
                where_clause = equivalent_condition_to_str(condition)
                
                # 构建SQL
                update_sql = f"""
                UPDATE {table_name}
                SET {set_clause}
                WHERE {where_clause}
                """
                
                # 合并参数
                params = {**updates}
                if condition:
                    for k, v in condition.items():
                        if isinstance(v, list):
                            for i, val in enumerate(v):
                                params[f"{k}_{i}"] = val
                        else:
                            params[k] = v
                
                # 执行更新
                cursor.execute(update_sql, params)
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Failed to update table {table_name}: {str(e)}")
            return False


    def delete(
            self,
            condition: Dict[str, Any],
            indexName: str,
            knowledgebaseId: str
    ) -> bool:
        """删除文档"""
        try:
            table_name = f"{indexName}_{knowledgebaseId}"
            with self.conn_pool.acquire() as conn:
                cursor = conn.cursor()
                
                # 构建WHERE子句
                #logger.debug(f"Oracle delete condition: {condition}")
                where_clause = equivalent_condition_to_str(condition)
                
                # 构建SQL
                delete_sql = f"""
                DELETE FROM {table_name}
                WHERE {where_clause}
                """
                
                # 执行删除
                logger.debug(f"Oracle delete sql: {delete_sql}")
                cursor.execute(delete_sql)
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Failed to delete from table {table_name}: {str(e)}")
            return False
        
    def sql(self, sql: str, fetch_size: int, format: str = "json"):
        """执行SQL查询"""
        logger.debug(f"OracleConnection.sql get sql: {sql}")
        
        try:
            # 清理SQL语句
            sql = re.sub(r"[ `]+", " ", sql)
            sql = sql.replace("%", "")
            
            # 处理特殊字段的LIKE查询
            replaces = []
            for r in re.finditer(r" ([a-z_]+_l?tks)( like | ?= ?)'([^']+)'", sql):
                fld, v = r.group(1), r.group(3)
                # Oracle中使用CONTAINS进行全文检索
                match = " CONTAINS({}, '{}') > 0 ".format(
                    fld, rag_tokenizer.fine_grained_tokenize(rag_tokenizer.tokenize(v)))
                replaces.append(
                    ("{}{}'{}'".format(
                        r.group(1),
                        r.group(2),
                        r.group(3)),
                    match))

            # 替换SQL中的条件
            for p, r in replaces:
                sql = sql.replace(p, r, 1)
            logger.debug(f"OracleConnection.sql to oracle: {sql}")

            # 执行SQL查询
            with self.conn_pool.acquire() as conn:
                cursor = conn.cursor()
                
                # 设置fetch size
                cursor.arraysize = fetch_size
                
                # 执行查询
                cursor.execute(sql)
                
                # 处理返回结果
                if format == "json":
                    columns = [col[0] for col in cursor.description]
                    results = []
                    for row in cursor:
                        results.append(dict(zip(columns, row)))
                    return results
                else:
                    # 其他格式直接返回原始数据
                    return cursor.fetchall()
                    
        except Exception as e:
            logger.error(f"Failed to execute SQL in Oracle: {str(e)}")
            return None
    
    def getTotal(self, res: tuple[list[dict], int] | list[dict]) -> int:
        """
        获取结果总数
        Args:
            res: 查询结果，可以是元组（结果列表，总数）或结果列表
        Returns:
            结果总数
        """
        if isinstance(res, tuple):
            return res[1]
        return len(res)

    def getChunkIds(self, res: tuple[list[dict], int] | list[dict]) -> list[str]:
        """
        获取所有chunk的ID列表
        Args:
            res: 查询结果，可以是元组（结果列表，总数）或结果列表
        Returns:
            chunk ID列表
        """
        if isinstance(res, tuple):
            res = res[0]
        return [row["id"] for row in res]

    def getFields(self, res: tuple[list[dict], int] | list[dict], fields: list[str]) -> dict[str, dict]:
        """
        获取指定字段的值
        Args:
            res: 查询结果，可以是元组（结果列表，总数）或结果列表
            fields: 需要获取的字段列表
        Returns:
            字典，key为chunk ID，value为包含指定字段的字典
        """
        if isinstance(res, tuple):
            res = res[0]
            
        res_fields = {}
        if not fields:
            return {}
            
        for row in res:
            id = row["id"]
            m = {"id": id}
            for fieldnm in fields:
                if fieldnm not in row:
                    m[fieldnm] = None
                    continue
                    
                v = row[fieldnm]
                # 处理特殊字段
                if fieldnm in ["important_kwd", "question_kwd", "entities_kwd", "tag_kwd", "source_id"]:
                    if isinstance(v, str):
                        m[fieldnm] = [kwd for kwd in v.split("###") if kwd]
                    else:
                        m[fieldnm] = []
                elif fieldnm == "position_int":
                    if isinstance(v, str) and v:
                        arr = [int(hex_val, 16) for hex_val in v.split('_')]
                        m[fieldnm] = [arr[i:i + 5] for i in range(0, len(arr), 5)]
                    else:
                        m[fieldnm] = []
                elif fieldnm in ["page_num_int", "top_int"]:
                    if isinstance(v, str) and v:
                        m[fieldnm] = [int(hex_val, 16) for hex_val in v.split('_')]
                    else:
                        m[fieldnm] = []
                else:
                    m[fieldnm] = str(v) if not isinstance(v, str) else v
                    
            res_fields[id] = m
            
        return res_fields
    
    def getHighlight(self, res: tuple[list[dict], int] | list[dict], keywords: list[str], fieldnm: str) -> dict[str, str]:
        """
        获取高亮结果
        Args:
            res: 查询结果，可以是元组（结果列表，总数）或结果列表
            keywords: 需要高亮的关键词列表
            fieldnm: 需要高亮的字段名
        Returns:
            字典，key为chunk ID，value为高亮后的文本
        """
        if isinstance(res, tuple):
            res = res[0]
            
        ans = {}
        if not res or fieldnm not in res[0]:
            return {}
            
        for row in res:
            id = row["id"]
            txt = row.get(fieldnm, "")
            if not txt:
                continue
                
            # 处理换行符
            txt = re.sub(r"[\r\n]", " ", txt, flags=re.IGNORECASE | re.MULTILINE)
            txts = []
            
            # 按句子分割
            for t in re.split(r"[.?!;\n]", txt):
                # 对每个关键词进行高亮处理
                for w in keywords:
                    t = re.sub(
                        r"(^|[ .?/'\"\(\)!,:;-])(%s)([ .?/'\"\(\)!,:;-])"
                        % re.escape(w),
                        r"\1<em>\2</em>\3",
                        t,
                        flags=re.IGNORECASE | re.MULTILINE,
                    )
                
                # 只保留包含高亮内容的句子
                if not re.search(r"<em>[^<>]+</em>", t, flags=re.IGNORECASE | re.MULTILINE):
                    continue
                    
                txts.append(t)
                
            ans[id] = "...".join(txts)
            
        return ans

    def getAggregation(self, res: tuple[list[dict], int] | list[dict], fieldnm: str) -> list:
        """
        获取聚合结果（暂未实现）
        Args:
            res: 查询结果，可以是元组（结果列表，总数）或结果列表
            fieldnm: 需要聚合的字段名
        Returns:
            空列表（功能未实现）
        """
        return list()

    def close(self):
        """关闭连接池"""
        if self.conn_pool:
            self.conn_pool.close()
            logger.info("Oracle connection pool closed")
