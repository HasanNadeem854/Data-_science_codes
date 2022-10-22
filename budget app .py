#!/usr/bin/env python
# coding: utf-8

# In[118]:


class Budget:
    def __init__ (self,categroy):
        self.categroy = categroy
        self.total = 0
        self.ledger = []
        
    def __repr__(self):
        header = self.categroy.center(30,"*")
        print(header)
        for item in self.ledger:
            amount = "%.2f" % item["amount"]
            desc = (item["description"][:25 - int(len(amount))] + "--") if len(item["description"])>28-len(amount) else item["description"] 
            a = len(amount)
            spaces = "_ " * int(30 -(len(amount))-len(desc))
            text = f"{desc}{spaces}{amount}"
            print (text)
        print("total is: " +  "%.2f" % self.total) 
                            
            
            
        
    def deposit(self,amount,description = ''):
        self.total += amount
        self.ledger.append({"amount":amount,"description":description})
        
    def withdrawal(self,amount,description = ""):
        if self.check_funds(amount):
            self.total -= amount
            self.ledger.append({"amount": - amount,"description":description})
        else:
            False
            
    def get_balance(self):
        return self.total
    
    def transfer(self,amount,instance):
        if self.check_funds(amount):
            self.total -= amount 
            self.ledger.append({"amount":-amount,"description":"transfer to " + instance.categroy})
            instance.total  += amount
            instance.ledger.append({"amount":amount,"description":"transfer from " + self.categroy})
        else:
            False
            
    def check_funds(self,amount):
        if (amount < self.total):
            return True
        else:
            return False        
    


# In[119]:


fun = Budget("food")
car = Budget("car")
fun.deposit(1000, "gujjer")
fun.withdrawal(100, "gujj")
fun.transfer(500,car)
fun.transfer(500,car)
print(fun.__repr__())


# In[14]:


catgs = ['FOOD','AUTO',"CLOTHINGS"]
chart = "Percentage spent by catogrey" + "\n"
percentages = [ 10 , 70 , 30]
height =  len(max(catgs, key = len))
new_catgs = [names.ljust(height) for names in catgs]
for num in reversed(range(0,110,10)):
  chart += f'{str(num) + "|" :>4}'
  for percentage in percentages:
    if percentage >= num:
      chart += "0 "
    else:
      chart += "  "
  chart += "\n"    
chart +="    " + "-" * 10 + "\n"

for name in zip(*new_catgs):
    chart += "     " + ('  '.join(name)) + "\n"
print(chart) 


# In[ ]:





# In[ ]:




