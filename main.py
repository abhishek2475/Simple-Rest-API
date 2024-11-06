
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, EmailStr
from typing import List
import requests
import threading
import logging
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

logging.basicConfig(level=logging.INFO)
app = FastAPI()

students_db = {}
lock = threading.Lock()  

class Student(BaseModel):
    id: int
    name: str
    age: int
    email: EmailStr

@app.post("/students", response_model=Student)
def create_student(student: Student):
    with lock:
        if student.id in students_db:
            raise HTTPException(status_code=400, detail="Student with this ID already exists.")
        students_db[student.id] = student
    return student


@app.get("/students", response_model=List[Student])
def get_all_students():
    with lock:
        return list(students_db.values())


@app.get("/students/{id}", response_model=Student)
def get_student(id: int):
    student = students_db.get(id)
    if not student:
        raise HTTPException(status_code=404, detail="Student not found.")
    return student


@app.put("/students/{id}", response_model=Student)
def update_student(id: int, student: Student):
    with lock:
        if id not in students_db:
            raise HTTPException(status_code=404, detail="Student not found.")
        students_db[id] = student
    return student


@app.delete("/students/{id}")
def delete_student(id: int):
    with lock:
        if id not in students_db:
            raise HTTPException(status_code=404, detail="Student not found.")
        del students_db[id]
    return {"message": "Student deleted successfully"}


@app.get("/students/{id}/summary")
def generate_student_summary(id: int):
    student = students_db.get(id)
    logging.info(students_db)
    if not student:
        raise HTTPException(status_code=404, detail="Student not found.")
    
    
    try:
        # response = requests.post(
        #     "http://127.0.0.1:11434/api/generate",
        #     json={"model": "llama3.2", "prompt": prompt}
        # )
        # response.raise_for_status()
        
        # Log the raw response text for debugging
        
        template = """Summarize this student profile using only the provided details. Be brief, accurate, and creative:

Profile:

Name: {name}
Age:{age}
Email:{email}
"""

        prompt = ChatPromptTemplate.from_template(template)

        model = OllamaLLM(model="llama3.2")

        chain = prompt | model

        summary=chain.invoke({"id":student.id,"name":student.name, "age":student.age, "email":student.email})
        
        # Try parsing the JSON response
    except requests.JSONDecodeError as e:
        logging.error(f"JSON parsing error: {e}")
        raise HTTPException(status_code=500, detail="Received invalid JSON from Ollama API.")
    except requests.RequestException as e:
        logging.error(f"Failed to connect to Ollama API: {e}")
        raise HTTPException(status_code=500, detail="Failed to connect to Ollama API.")
    
    return {"summary": summary}
@app.get("/test-ollama")
def test_ollama():
    try:
        response = requests.post(
            "http://127.0.0.1:11434/api/generate",
            json={"model": "llama3.2", "prompt": "2+2"}
        )
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print("Error Response Content:", e.response.content if e.response else "No response content")
        raise HTTPException(status_code=500, detail="Failed to connect to Ollama API.")
