import os
import sys

import sqlalchemy
from dotenv import load_dotenv, find_dotenv

from sqlalchemy import create_engine, text

sys.path.append('../../../..')

# read local .env file
_ = load_dotenv(find_dotenv())
ORACLE_DB_CONNECT_STRING = os.environ['ORACLE_DB_CONNECT_STRING']

# print(sqlalchemy.__version__)
# 创建引擎
engine = create_engine(ORACLE_DB_CONNECT_STRING, echo=False)

with engine.connect() as conn:
    try:
        result = conn.execute(text("select '========= connect database successfully ========='"))
        print(result.all()[0][0])
        conn.commit()
    except Exception as e:
        print(f"=========\n {e} \n=========")
        conn.rollback()
