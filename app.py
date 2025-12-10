from fastapi import FastAPI, WebSocket,File, UploadFile, Form, HTTPException

from src.websocket.ppe_w_local1 import run_ppe_detection

# n


from src.handlers.ppe_handler import ppe_websocket_handler

from fastapi.middleware.cors import CORSMiddleware


from concurrent.futures import ThreadPoolExecutor

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- Session Stores ----------------
ppe_sessions = {}


detection_executor = ThreadPoolExecutor(max_workers=10)
storage_executor = ThreadPoolExecutor(max_workers=5)


#--------------------------------------------------------------------------- WebSocket for all Models ------------------------------------------------------------------------------#



# ---------------- PPE WebSocket ----------------
@app.websocket("/ws/ppe/{client_id}")
async def websocket_ppe(ws: WebSocket,client_id: str):
    await ppe_websocket_handler(detection_executor, storage_executor, ws,client_id, ppe_sessions, run_ppe_detection, "PPE")



