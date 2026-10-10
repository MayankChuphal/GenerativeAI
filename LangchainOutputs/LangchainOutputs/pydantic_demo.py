from pydantic import BaseModel

class Student(BaseModel):
    name:str
    
    
new_student:Student={
    'name':25
}
## no validation for this format as PYdantic expects arguements in name='Mayank' format

demo_student={'name':'Mayank'}

student = Student(**demo_student)

print(new_student)
print(student)