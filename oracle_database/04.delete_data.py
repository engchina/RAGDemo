import os
import sys

import sqlalchemy
from dotenv import load_dotenv, find_dotenv

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

sys.path.append('../../..')

# read local .env file
_ = load_dotenv(find_dotenv())
ORACLE_DB_CONNECT_STRING = os.environ['ORACLE_DB_CONNECT_STRING']

# print(sqlalchemy.__version__)
# 创建引擎
engine = create_engine(ORACLE_DB_CONNECT_STRING, echo=False)

delete_stmt = text("DELETE from daily_task_report")
select_stmt = text("SELECT employee_name, task_report FROM daily_task_report")
with Session(engine) as session:
    try:
        print("========= select data before delete =========")
        result = session.execute(select_stmt)
        original_row_count = 0
        for row in result:
            original_row_count += 1
            print(f"employee_name: {row.employee_name}, task_report: {row.task_report}")

        result = session.execute(delete_stmt)
        # print(result.rowcount)
        if result.rowcount == original_row_count:
            print("delete data successfully")
        else:
            print("delete data failed")

        print("========= select data after delete =========")
        result = session.execute(select_stmt)
        for row in result:
            print(f"employee_name: {row.employee_name}, task_report: {row.task_report}")
        session.commit()
    except Exception as e:
        print(f"=========\n {e} \n=========")
        session.rollback()
