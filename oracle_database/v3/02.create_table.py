import os
import sys

import sqlalchemy
from dotenv import load_dotenv, find_dotenv

from sqlalchemy import create_engine, text

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

with engine.connect() as conn:
    try:
        result = conn.execute(text("""
CREATE TABLE "EMPLOYEE" 
(	
"NAME" VARCHAR2(50 BYTE), 
"DEPARTMENT_NAME" VARCHAR2(50 BYTE), 
"HIRE_DATE" DATE,
"BIRTHDAY" DATE, 
"SALARY" NUMBER, 
"ADDRESS" VARCHAR2(200 BYTE),
"PASSWORD" VARCHAR2(20 BYTE), 
"ROLE" VARCHAR2(20 BYTE), 
"VECTOR_FLAG" CHAR(1 BYTE)
)        
        """))
        # print(result.rowcount)

        if result.rowcount == 0:
            print("========= create table successfully =========")
            result = conn.execute(text(""" COMMENT ON COLUMN "EMPLOYEE"."NAME" IS '名前' """))
            result = conn.execute(text(""" COMMENT ON COLUMN "EMPLOYEE"."DEPARTMENT_NAME" IS '部署' """))
            result = conn.execute(text(""" COMMENT ON COLUMN "EMPLOYEE"."HIRE_DATE" IS '入社日' """))
            result = conn.execute(text(""" COMMENT ON COLUMN "EMPLOYEE"."BIRTHDAY" IS '誕生日' """))
            result = conn.execute(text(""" COMMENT ON COLUMN "EMPLOYEE"."SALARY" IS '給料' """))
            result = conn.execute(text(""" COMMENT ON COLUMN "EMPLOYEE"."ADDRESS" IS '住所' """))
            result = conn.execute(text(""" COMMENT ON COLUMN "EMPLOYEE"."PASSWORD" IS 'パスワード' """))
            result = conn.execute(text(""" COMMENT ON COLUMN "EMPLOYEE"."ROLE" IS 'ロール' """))
            result = conn.execute(text(""" COMMENT ON COLUMN "EMPLOYEE"."VECTOR_FLAG" IS 'ベクトル化フラグ' """))
            conn.commit()
        else:
            print("========= create table failed =========")
            conn.rollback()
    except Exception as e:
        print(f"=========\n {e} \n=========")
        conn.rollback()
