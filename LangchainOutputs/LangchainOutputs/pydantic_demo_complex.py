from pydantic import BaseModel,EmailStr,Field
from typing import Optional

class Student(BaseModel):
    name:str = 'Ayush'
    age:Optional[int] = None
    email:EmailStr
    cgpa:float = Field(default=5,gt=0,lt=10,description='This is the cgpa of the student')
    


demo_student={'age':'32','email':'abc@gmail.com','cgpa':9.9}

student = Student(**demo_student)

student_json = student.model_dump_json()
student_dict = dict(student);



print(student)

print(student_json)
print(student_dict)