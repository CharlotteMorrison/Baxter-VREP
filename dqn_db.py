import sqlite3
from sqlite3 import Error
import io
import numpy as np


def create_connection(db_file):
    conn = None
    try:
        conn = sqlite3.connect(db_file, detect_types=sqlite3.PARSE_DECLTYPES)
        return conn
    except Error as e:
        print(e)
    return conn


def adapt_array(arr):
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())


def convert_array(text):
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out, allow_pickle=True)


def insert_mem(conn, row):
    sql = ''' INSERT INTO Mem_Replay(action,reward,state,new_state,done) VALUES(?,?,?,?,?) '''
    try:
        cur = conn.cursor()
        cur.execute(sql, row)
        cur.close()
    except Error as e:
        print(e)


def insert_freq(conn, row):
    sql = ''' INSERT INTO Frequency(action,reward) VALUES(?,?) '''
    cur = conn.cursor()
    cur.execute(sql, row)
    return cur.lastrowid


def get_all_memory(conn):
    sql = 'SELECT state, action, new_state, reward, done FROM Mem_Replay;'

    cur = conn.cursor()
    cur.execute(sql)
    return cur.fetchall()


def get_frequency(conn, action):
    cur = conn.cursor()
    cur.execute("SELECT MAX(reward) FROM Frequency WHERE action=?", (action,))

    temp_reward = cur.fetchall()
    temp_reward = temp_reward[0][0]

    if temp_reward == None:
        return 0, 0

    else:
        cur = conn.cursor()
        cur.execute("SELECT * FROM Frequency WHERE action=? AND reward=?", (action, temp_reward,))

        rewards = cur.fetchall()

        reward_frequency = len(rewards)

        return temp_reward, reward_frequency


# Converts np.array to TEXT when inserting
sqlite3.register_adapter(np.ndarray, adapt_array)

# Converts TEXT to np.array when selecting
sqlite3.register_converter("array", convert_array)
