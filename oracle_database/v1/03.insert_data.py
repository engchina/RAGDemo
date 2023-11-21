import os
import sys

import sqlalchemy
from dotenv import load_dotenv, find_dotenv

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

sys.path.append('../../../..')

# read local .env file
_ = load_dotenv(find_dotenv())
ORACLE_DB_CONNECT_STRING = os.environ['ORACLE_DB_CONNECT_STRING']

# print(sqlalchemy.__version__)
# 创建引擎
engine = create_engine(ORACLE_DB_CONNECT_STRING, echo=False)

insert_stmt = text("INSERT INTO daily_task_report (employee_name, task_report) VALUES (:employee_name, :task_report)")
select_stmt = text("SELECT employee_name, task_report FROM daily_task_report")
with Session(engine) as session:
    try:
        print("========= select data before insert =========")
        result = session.execute(select_stmt)
        before_row_count = 0
        for row in result:
            before_row_count += 1
            print(f"employee_name: {row.employee_name}, task_report: {row.task_report}")

        result = session.execute(insert_stmt, {"employee_name": "田中", "task_report": "仕事をする"})
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
            print(f"employee_name: {row.employee_name}, task_report: {row.task_report}")
        if after_row_count == before_row_count + 1:
            print("insert data successfully")
        else:
            print("insert data failed")
        session.commit()
    except Exception as e:
        print(f"=========\n {e} \n=========")
        session.rollback()
