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

insert_stmt = text("""
INSERT INTO employee (name, department_name, hire_date, birthday, salary, address, password, role, vector_flag) 
VALUES (:name, :department_name, :hire_date,:birthday, :salary, :address, :password, :role, :vector_flag)
""")
select_stmt = text("SELECT name, department_name, hire_date, birthday, salary, "
                   "address, password, role, vector_flag FROM employee")
with Session(engine) as session:
    try:
        print("========= select data before insert =========")
        session.execute(text("ALTER SESSION SET NLS_DATE_FORMAT = 'YYYY-MM-DD'"))
        result = session.execute(select_stmt)
        before_row_count = 0
        for row in result:
            before_row_count += 1
            print(
                f"name: {row.name}, department_name: {row.department_name}, hire_date: {row.hire_date}, "
                f"birthday: {row.birthday}, salary: {row.salary}, address: {row.address}, password: {row.password}, "
                f"role: {row.role}, vector_flag: {row.vector_flag}")

        result = session.execute(insert_stmt, {"name": "人事太郎", "department_name": "人事", "hire_date": "2022-01-01",
                                               "birthday": "1980-01-01", "salary": 30, "address": "東京",
                                               "password": "123456", "role": "admin", "vector_flag": "Y"})
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
                f"name: {row.name}, department_name: {row.department_name}, hire_date: {row.hire_date}, "
                f"birthday: {row.birthday}, salary: {row.salary}, address: {row.address}, password: {row.password}, "
                f"role: {row.role}, vector_flag: {row.vector_flag}")
        if after_row_count == before_row_count + 1:
            print("insert data successfully")
        else:
            print("insert data failed")
        session.commit()
    except Exception as e:
        print(f"=========\n {e} \n=========")
        session.rollback()
