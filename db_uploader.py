import sqlalchemy
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy import Table, Column, Integer, String, Float, MetaData
from sqlalchemy.orm import declarative_base
from sqlalchemy import insert

# Importing income data

df_income = pd.read_csv("./sources/income.csv")
df_income = df_income.reset_index()
df_income = df_income.rename(columns = {'index': 'pk'})
df_income

# Importing partners data

df_partners = pd.read_csv("./sources/partners_small.csv")
df_partners = df_partners.reset_index()
df_partners = df_partners.rename(columns = {'index': 'pk'})
df_partners

# Defining sqlalchemy engine

engine = create_engine('sqlite:///L:\\Projects\\22005 - Housing Needs Assessment\\Scripts\\Dashboard\\dashboard_prototype_kook_practice\\sources\\hart.db')#, echo=True)

# Creating tables

Base = declarative_base()

class Partners(Base):
    __tablename__ = "partners"
    
    # define your primary key
    pk = Column(Integer, primary_key=True, comment='primary key')

    # columns except pk
    Geography = Column(String)
    for i in df_partners.columns[2:]:
        vars()[f'{i}'] = Column(Float)

Partners.__table__.create(bind=engine, checkfirst=True)

class Income(Base):
    __tablename__ = "income"
    
    # define your primary key
    pk = Column(Integer, primary_key=True, comment='primary key')

    # columns except pk
    for i in df_income.columns[1:]:
        if df_income.dtypes[i] =='int64':
            vars()[f'{i}'] = Column(Integer)
        else:
            vars()[f'{i}'] = Column(String)

Income.__table__.create(bind=engine, checkfirst=True)

# Inserting data

conn = engine.connect()

for i in range(0, len(df_partners)):
    conn.execute(insert(Partners), [df_partners.loc[i,:].to_dict()])

for i in range(0, len(df_income)):
    conn.execute(insert(Income), [df_income.loc[i,:].to_dict()])

conn.close()