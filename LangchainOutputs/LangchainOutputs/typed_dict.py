from typing import TypedDict

class Person(TypedDict):
    name:str
    age:int
    
demo_dict: Person = {
    'name':'Mayank Chuphal',
    'age':24
}

print(demo_dict)