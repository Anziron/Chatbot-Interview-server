import os, sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver
database_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),"checkpoints.db"))
class Database:
    _instance = None
    conn = sqlite3.connect(database_path, check_same_thread=False)
    memory = SqliteSaver(conn)
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(Database, cls).__new__(cls, *args, **kwargs)
        return cls.memory
    
    @classmethod
    def delete_thread(cls, thread_id):
        """删除指定线程ID的所有记忆数据"""
        cursor = cls.conn.cursor()
        try:
            # 删除checkpoints表中的线程数据
            cursor.execute("DELETE FROM checkpoints WHERE thread_id = ?", (thread_id,))
            # 删除checkpoint_writes表中的线程数据
            cursor.execute("DELETE FROM checkpoint_writes WHERE thread_id = ?", (thread_id,))
            # 删除checkpoint_blobs表中的线程数据（如果存在）
            cursor.execute("DELETE FROM checkpoint_blobs WHERE thread_id = ?", (thread_id,))
            cls.conn.commit()
            return True
        except Exception as e:
            cls.conn.rollback()
            raise e
        finally:
            cursor.close()

checkpointer = Database()
    







