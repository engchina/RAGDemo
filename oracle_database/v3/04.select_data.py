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
MY_ADB_ADMIN_USER = os.environ['MY_ADB_ADMIN_USER']
MY_ADB_APP_USER = os.environ['MY_ADB_APP_USER']
MY_ADB_SERVICE_NAME = os.environ['MY_ADB_SERVICE_NAME']
MY_ATP_CONFIG_DIR = os.environ['MY_ATP_CONFIG_DIR']
MY_ATP_WALLET_LOCATION = os.environ['MY_ATP_WALLET_LOCATION']
MY_ADB_WALLET_PASSWORD = os.environ['MY_ADB_WALLET_PASSWORD']
MY_ADW_ADMIN_PASSWORD = os.environ['MY_ADW_ADMIN_PASSWORD']
MY_ATP_ADMIN_PASSWORD = os.environ['MY_ATP_ADMIN_PASSWORD']

# print(sqlalchemy.__version__)
# 创建引擎
# engine = create_engine(ORACLE_DB_CONNECT_STRING, echo=False)
engine = create_engine(f'oracle+oracledb://:@',
                       connect_args={
                           "user": MY_ADB_APP_USER,
                           "password": MY_ATP_ADMIN_PASSWORD,
                           "dsn": MY_ADB_SERVICE_NAME,
                           "config_dir": MY_ATP_CONFIG_DIR,
                           "wallet_location": MY_ATP_WALLET_LOCATION,
                           "wallet_password": MY_ADB_WALLET_PASSWORD,
                       }, echo=False)

select_stmt = text("SELECT name, department_name, hire_date, birthday, salary, "
                   "address, password, role, vector_flag FROM employee")
with Session(engine) as session:
    try:
        print("========= select data =========")
        session.execute(text("ALTER SESSION SET NLS_DATE_FORMAT = 'YYYY-MM-DD'"))
        result = session.execute(select_stmt)
        before_row_count = 0
        for row in result:
            before_row_count += 1
            print(
                f"name: {row.name}, department_name: {row.department_name}, hire_date: {row.hire_date}, "
                f"birthday: {row.birthday}, salary: {row.salary}, address: {row.address}, password: {row.password}, "
                f"role: {row.role}, vector_flag: {row.vector_flag}")
        session.commit()
    except Exception as e:
        print(f"=========\n {e} \n=========")
        session.rollback()
