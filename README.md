# RAGDemo
RAG Demo

## 準備

```
conda create -n ragdemo python=3.10 -y
conda activate ragdemo
```

```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
conda install -y -c "nvidia/label/cuda-12.1.0" cuda-runtime

python -c "import torch;print(torch.cuda.is_available());"
---
True
```

```
pip install -r requirements.txt
```

## RAG デモの起動

```
python app.py
```

ブラウザーで [http://localhost:7860](http://localhost:7860) を開いて、アクセスしてください。

refer: [https://python.langchain.com/docs/use_cases/question_answering/](https://python.langchain.com/docs/use_cases/question_answering/)


## （Optional） PGVector

PGVector の起動、

```
mkdir -p /root/data/pg; chmod 777 /root/data/pg
docker run --name pgvector --restart=always -p 5432:5432 -v /root/data/pg:/var/lib/postgresql/data:rw -e POSTGRES_INITDB_ARGS="--locale-provider=icu --icu-locale=ja-x-icu" -e POSTGRES_USER=postgres -e POSTGRES_PASSWORD=password -e POSTGRES_DB=postgres -d ankane/pgvector:v0.5.1
```

拡張機能を有効にする、

```
docker exec -it pgvector bash
psql -U postgres -d postgres
CREATE EXTENSION vector;
```

データベースの作成、

```
docker exec -it pgvector bash
psql -U postgres -d postgres
CREATE DATABASE ragdemo;
```

[pgvector document](https://github.com/pgvector/pgvector)

## (Optional) Oracle Database

refer: [https://docs.sqlalchemy.org/en/20/index.html](https://docs.sqlalchemy.org/en/20/index.html)

```
# grant dba to pdbadmin;
```

```
docker exec -it oracledb23c bash
export NLS_LANG=Japanese_Japan.AL32UTF8;
sqlplus pdbadmin/<your_password>@FREEPDB1
```

```
select * from USER_TABLES;

select * from USER_TAB_COMMENTS;

select * from USER_COL_COMMENTS;
```

```
select T1.TABLE_NAME, T2.COMMENTS as TAB_COMMNETS, T3.COLUMN_NAME as COLUMN_NAME, T3.COMMENTS as COL_COMMNETS
from USER_TABLES T1, USER_TAB_COMMENTS T2, USER_COL_COMMENTS T3
where T1.TABLE_NAME = T2.TABLE_NAME
and T1.TABLE_NAME = T3.TABLE_NAME;
```

## (Optional) Oracle Autonomous Database Free

refer: [Oracle Autonomous Database Free](https://container-registry.oracle.com/ords/f?p=113:4:107533797608756:::4:P4_REPOSITORY,AI_REPOSITORY,AI_REPOSITORY_NAME,P4_REPOSITORY_NAME,P4_EULA_ID,P4_BUSINESS_AREA_ID:2223,2223,Oracle%20Autonomous%20Database%20Free,Oracle%20Autonomous%20Database%20Free,1,0&cs=3RnJw905pdwNm03uI1VmNmJqXLDrZcxJs_StSOyTtDWB3zituLMGmojIKIsGEK51Q3XjFtcA3SBOcxAOGjsJ58g)

Install: 

```
mkdir -p /root/data/adb; chmod 777 /root/data/adb
docker run -d \
-p 1521:1522 \
-p 1522:1522 \
-p 8443:8443 \
-p 27017:27017 \
-e MY_ADB_WALLET_PASSWORD=*** \
-e MY_ADW_ADMIN_PASSWORD=*** \
-e MY_ATP_ADMIN_PASSWORD=*** \
--cap-add SYS_ADMIN \
--device /dev/fuse \
--name adb-free \
--volume /root/data/adb:/u01/data \
container-registry.oracle.com/database/adb-free:23.10.2.2
```

Connecting to Oracle Autonomous Database Free container:

| Application  | MY_ATP  | MY_ADW  |
|---|---|---|
| APEX  | https://localhost:8443/ords/my_atp/  | https://localhost:8443/ords/my_adw/  |
| Database Actions	  | https://localhost:8443/ords/my_atp/sql-developer  | https://localhost:8443/ords/my_adw/sql-developer  |

Wallet Setup:

In the container, TLS wallet is generated at location `/u01/app/oracle/wallets/tls_wallet`,

```
docker cp adb-free:/u01/app/oracle/wallets/tls_wallet /root/tls_wallet
```

Point TNS_ADMIN environment variable to the wallet directory

```
export TNS_ADMIN=/root/tls_wallet
```


If you want to connect to a remote host where the ADB free container is running, replace localhost in $TNS_ADMIN/tnsnames.ora with the remote host FQDN,

```
sed -i 's/localhost/192.168.31.15/g' $TNS_ADMIN/tnsnames.ora
```

MY_ATP TNS aliases:
```
For mTLS use the following:
my_atp_medium
my_atp_high
my_atp_low
my_atp_tp
my_atp_tpurgent

For TLS use the following:
my_atp_medium_tls
my_atp_high_tls
my_atp_low_tls
my_atp_tp_tls
my_atp_tpurgent_tls
```

MY_ADW TNS aliases:

```
For mTLS use the following:
my_adw_medium
my_adw_high
my_adw_low

For TLS use the following:
my_adw_medium_tls
my_adw_high_tls
my_adw_low_tls
```

Python thin driver:

```
pip install oracledb

import oracledb
dsn = "admin/<my_adw_admin_password>@my_adw_medium"
conn = oracledb.connect(dsn=dsn, wallet_location="/root/tls_wallet", wallet_password="***")
cr = conn.cursor()
r = cr.execute("SELECT 1 FROM DUAL")
print(r.fetchall())

>> [(1,)]
```

Create an app user:

```
sqlplus admin/<my_atp_admin_password>@my_atp_medium

CREATE USER APP_USER IDENTIFIED BY "<my_app_user_password>" QUOTA UNLIMITED ON DATA;

-- ADD ROLES
GRANT CONNECT TO APP_USER;
GRANT CONSOLE_DEVELOPER TO APP_USER;
GRANT DWROLE TO APP_USER;
GRANT RESOURCE TO APP_USER;  


-- ENABLE REST
BEGIN
    ORDS.ENABLE_SCHEMA(
        p_enabled => TRUE,
        p_schema => 'APP_USER',
        p_url_mapping_type => 'BASE_PATH',
        p_url_mapping_pattern => 'app_user',
        p_auto_rest_auth=> TRUE
    );
    commit;
END;
/

-- QUOTA
ALTER USER APP_USER QUOTA UNLIMITED ON DATA;
```

refer: [Using SQLAlchemy 2.0 with python-oracledb for Oracle Database](https://medium.com/oracledevs/using-the-development-branch-of-sqlalchemy-2-0-with-python-oracledb-d6e89090899c)

refer: [Connect Python Applications with a Wallet (mTLS)](https://docs.oracle.com/en-us/iaas/autonomous-database-serverless/doc/connecting-python-mtls.html#GUID-C7D9BDA0-7147-4089-A87E-F9DBB126C6F1)