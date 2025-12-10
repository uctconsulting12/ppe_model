# import json
# import logging
# import psycopg2
# import os
# from dotenv import load_dotenv

# logger = logging.getLogger("detection")
# logger.setLevel(logging.INFO)

# # Load environment variables
# load_dotenv()

# # PostgreSQL connection
# try:
#     conn = psycopg2.connect(
#         host=os.environ["DB_HOST"],
#         dbname=os.environ["DB_NAME"],
#         user=os.environ["DB_USER"],
#         password=os.environ["DB_PASSWORD"],
#         port=int(os.environ.get("DB_PORT", 5432))
#     )
#     conn.autocommit = True
#     cursor = conn.cursor()
#     logger.info("✅ Connected to PostgreSQL")
# except Exception as e:
#     logger.error(f"Failed to connect to PostgreSQL: {e}")
#     raise


# def insert_ppe_frame(data: dict, s3_url: str):
#     """
#     Insert PPE frame data into ppe_detections table.
#     Assumes table has a primary key 'id' (BIGSERIAL).
#     """
#     try:
#         with conn.cursor() as cursor:
#             insert_query = """
#                 INSERT INTO ppe_detections (
#                     s3_url, detections, user_id, org_id, camera_id, time_stamp, frame_num
#                 )
#                 VALUES (%s, %s, %s, %s, %s, %s, %s)
#                 RETURNING id;
#             """
#             cursor.execute(
#                 insert_query,
#                 (
#                     s3_url,
#                     json.dumps(data['detections']),
#                     data['user_id'],
#                     data['org_id'],
#                     data['camera_id'],
#                     data['time_stamp'],
#                     data['frame_num']
#                 )
#             )

#             inserted_id = cursor.fetchone()[0]
#             conn.commit()
#             logger.info(f"✅ PPE frame stored with id={inserted_id}, frame_num={data['frame_num']}")
#             return inserted_id
#     except Exception as e:
#         conn.rollback()
#         logger.error(f"❌ Failed to insert PPE frame: {e}")
#         return None




import os
import logging
from psycopg2.pool import SimpleConnectionPool
from dotenv import load_dotenv
import json
import logging

logger = logging.getLogger("detection")
load_dotenv()

DB_CONFIG = {
    "host": os.getenv("DB_HOST"),
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "port": int(os.getenv("DB_PORT", 5432)),
}

try:
    pool = SimpleConnectionPool(
        minconn=1,
        maxconn=20,
        **DB_CONFIG
    )
    logger.info("✅ PostgreSQL connection pool created")
except Exception as e:
    logger.error(f"❌ Failed to create connection pool: {e}")
    raise






logger = logging.getLogger("detection")


def insert_ppe_frame(data: dict, s3_url: str):
    """
    Insert PPE frame data using connection pool.
    """
    conn = None
    try:
        conn = pool.getconn()
        cursor = conn.cursor()

        insert_query = """
            INSERT INTO ppe_detections (
                s3_url, detections, user_id, org_id, camera_id, time_stamp, frame_num
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            RETURNING id;
        """

        cursor.execute(
            insert_query,
            (
                s3_url,
                json.dumps(data['detections']),
                data['user_id'],
                data['org_id'],
                data['camera_id'],
                data['time_stamp'],
                data['frame_num']
            )
        )

        inserted_id = cursor.fetchone()[0]
        conn.commit()
        cursor.close()

        logger.info(
            f"✅ PPE frame stored with id={inserted_id}, frame_num={data['frame_num']}"
        )

        return inserted_id

    except Exception as e:
        if conn:
            conn.rollback()
        logger.error(f"❌ Failed to insert PPE frame: {e}")
        return None

    finally:
        if conn:
            pool.putconn(conn)
