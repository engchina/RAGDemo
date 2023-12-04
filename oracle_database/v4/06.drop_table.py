import os
import sys

import sqlalchemy
from dotenv import load_dotenv, find_dotenv

from sqlalchemy import create_engine, text

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

with engine.connect() as conn:
    try:
        result = conn.execute(text("DROP TABLE employee"))
        # print(result.rowcount)
        if result.rowcount == 0:
            print("========= drop table successfully =========")
            conn.commit()
        else:
            print("========= drop table failed =========")
            conn.rollback()
    except Exception as e:
        print(f"=========\n {e} \n=========")
        conn.rollback()
