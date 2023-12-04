import os
import sys

import sqlalchemy
from dotenv import load_dotenv, find_dotenv

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

sys.path.append('../../..')

# read local .env file
_ = load_dotenv(find_dotenv())
MY_DB23C_APP_USER = os.environ['MY_DB23C_APP_USER']
MY_DB23C_DSN_NAME = os.environ['MY_DB23C_DSN_NAME']
MY_DB23C_PASSWORD = os.environ['MY_DB23C_PASSWORD']

# print(sqlalchemy.__version__)
# 创建引擎
# engine = create_engine(ORACLE_DB_CONNECT_STRING, echo=False)
engine = create_engine(f'oracle+oracledb://:@',
                       connect_args={
                           "user": MY_DB23C_APP_USER,
                           "password": MY_DB23C_PASSWORD,
                           "dsn": MY_DB23C_DSN_NAME
                       }, echo=False)

insert_stmt = text("""
INSERT INTO t1 (id, embeddings) 
VALUES (:id, :embeddings)
""")
"""
 (oracledb.exceptions.DatabaseError) ORA-01484: arrays can only be bound to PL/SQL statements
Help: https://docs.oracle.com/error-help/db/ora-01484/
"""
select_stmt = text("SELECT id, embeddings FROM t1")
with Session(engine) as session:
    try:
        print("========= select data before insert =========")
        session.execute(text("ALTER SESSION SET NLS_DATE_FORMAT = 'YYYY-MM-DD'"))
        result = session.execute(select_stmt)
        before_row_count = 0
        for row in result:
            before_row_count += 1
            print(
                f"id: {row.id}, embeddings: {row.embeddings}")

        result = session.execute(insert_stmt, {"id": 1, "embeddings": [1, 1, 1]})
        # print(result.rowcount)
        # if result.rowcount == 1:
        #     print("insert data successfully")
        # else:
        #     print("insert data failed")

        print("========= select data after insert =========")
        result = session.execute(select_stmt)
        after_row_count = 0
        for row in result:
            after_row_count += 1
            print(
                f"id: {row.id}, embeddings: {row.embeddings}")
        if after_row_count == before_row_count + 1:
            print("insert data successfully")
        else:
            print("insert data failed")
        session.commit()
    except Exception as e:
        print(f"=========\n {e} \n=========")
        session.rollback()
