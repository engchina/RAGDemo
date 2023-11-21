import os
import sys

import sqlalchemy
from dotenv import load_dotenv, find_dotenv

from sqlalchemy import create_engine, text

sys.path.append('../../../..')

# read local .env file
_ = load_dotenv(find_dotenv())
ORACLE_DB_CONNECT_STRING = os.environ['ORACLE_DB_CONNECT_STRING']

# 创建引擎
engine = create_engine(ORACLE_DB_CONNECT_STRING, echo=False)

with engine.connect() as conn:
    try:
        result = conn.execute(text("DROP TABLE daily_task_report"))
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
