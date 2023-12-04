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

update_stmt = text("""
UPDATE employee set vector_flag = 'Y'
WHERE vector_flag = 'N'
""")
select_stmt = text("SELECT name, department_name, hire_date, birthday, salary, "
                   "address, password, role, vector_flag FROM employee")
with Session(engine) as session:
    try:
        print("========= select data before update =========")
        session.execute(text("ALTER SESSION SET NLS_DATE_FORMAT = 'YYYY-MM-DD'"))
        result = session.execute(select_stmt)
        before_row_count = 0
        for row in result:
            before_row_count += 1
            print(
                f"name: {row.name}, department_name: {row.department_name}, hire_date: {row.hire_date}, "
                f"birthday: {row.birthday}, salary: {row.salary}, address: {row.address}, password: {row.password}, "
                f"role: {row.role}, vector_flag: {row.vector_flag}")

        result = session.execute(update_stmt)
        # print(result.rowcount)
        # if result.rowcount == 1:
        #     print("insert data successfully")
        # else:
        #     print("insert data failed")

        print("========= select data after update =========")
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
