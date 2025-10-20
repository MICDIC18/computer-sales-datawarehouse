#!/usr/bin/env python
# coding: utf-8

# # PRE-PROCESSING

# In[1]:


from datetime import datetime
start_time = datetime.now() # just to measure total execution time


# In[2]:


#function to get a dictionary with keys the column name in the header
#and as values the correspondeing columns (in a list format)

def dict_from_header(header, rows_ds):
    header_dict = header_idx(header) #get index from the header
    dict_columns = dict()

    #for every column get its index and extract its rows
    for key in header_dict.keys():
        idx = header_dict[key] #get index of col
        dict_columns[key] = extract_col(rows_ds, idx) #extract column's values
        
    return dict_columns

#function to get the index of the columns
def header_idx(header):
    #to retrieve the index of the column from its name
    header_dict = dict([(feat, n) for n, feat in enumerate(header)])
    return header_dict

#function to extract a single column from a list of lists by inputting the index
def extract_col(rows, idx):
    return [row[idx] for row in rows]


# In[3]:


#input the dataset in the dictionary format and the columns for which a table is requested
#multiple coluns are accepted, in order to allow creating tables with multiple columns

def unify_rows(dict_columns, *columns_to_merge):
    merging = list()
    
    #for every column to merge
    for col in columns_to_merge:
        row_of_col = dict_columns[col] #get the rows
        merging.append(row_of_col) #append the rows to a list
    
    return zip(*merging) #zip the list of lists

#to create a table from multiple columns
def gen_table(dict_columns, *columns):
    table_records = list(unify_rows(dict_columns, *columns)) #get the table of records
    return [columns] + table_records #returns the full table with the header and the records

#it is the same as before but it accepts a list of columns instead of multiple column parameters
def gen_table_from_list(dict_columns, columns):
    table_records = list(unify_rows(dict_columns, *columns))
    return [columns] + table_records

#it gets the table of distinct elements and it returns it sorted
def gen_table_distinct(dict_columns, *columns):
    #here a set is created, then it is made as a list in order to sort it
    table_records = sorted(list(set(unify_rows(dict_columns, *columns))))
    #finally the header is added
    return [columns] + table_records


# In[4]:


#input the table and the primary key column's name
#it adds the primary key to the table as its first column

def set_primary_key(table, name_primary_key):
    
    #get the primary key column (with header the input name) by ranging across the rows
    id_col = [name_primary_key] + [str(i) for i in range(1, len(table)+1)]
    #zipping the primary key and the previous table, it has to be unzipped though
    table_to_unzip = list(zip(id_col, table))
    #get the number of columns (length of a row)
    len_rows = len(table[0])
    
    #the output
    outres = list()

    #for each row in the table
    for row in table_to_unzip:
        #get the first element (the primary key)
        unzipped = [row[0]]
        
        #loop across the second element with the original columns
        for n in range(len_rows):
            #add the columns to the unzipped table
            unzipped.append(row[1][n])
            
        #add the current row to the output
        outres.append(tuple(unzipped))
        
    return outres


# In[5]:


import pandas as pd

def tuples_to_dataframe(data):
    # Ottieni i nomi delle colonne dalla prima tupla
    column_names = data[0]

    # Ottieni i dati escludendo la prima tupla
    data_rows = data[1:]

    # Costruisci il DataFrame utilizzando i nomi delle colonne e i dati
    df = pd.DataFrame(data_rows, columns=column_names)

    return df


# In[6]:


import pandas as pd
sales_df = pd.read_csv("C:\\Users\\HP\\Downloads\\Archivio\\computer_sales.csv", dtype={'time_code': str})
sales_df.shape


# In[7]:


sum(sales_df.duplicated())


# In[8]:


sales_df


# In[9]:


sales_df.isnull().sum()


# In[10]:


sales_df['ram_vendor_name'] = sales_df['ram_vendor_name'].replace("Mike's Computer Shop", "Mikes Computer Shop")
sales_df['gpu_vendor_name'] = sales_df['gpu_vendor_name'].replace("Mike's Computer Shop", "Mikes Computer Shop")
sales_df['cpu_vendor_name'] = sales_df['cpu_vendor_name'].replace("Mike's Computer Shop", "Mikes Computer Shop")


# # CREAZIONE TABELLA TIME

# In[11]:


import pandas as pd
import calendar

sales_df['day'] = sales_df['time_code'].str[-2:].astype(int)

# Creazione della variabile 'month'
sales_df['month'] = sales_df['time_code'].str[-4:-2].astype(int)

# Creazione della variabile 'year'
sales_df['year'] = sales_df['time_code'].str[:4].astype(int)

# Creazione della variabile 'quarter'
def get_quarter(month):
    if month in range(1, 4):
        return 'Q1'
    elif month in range(4, 7):
        return 'Q2'
    elif month in range(7, 10):
        return 'Q3'
    else:
        return 'Q4'

sales_df['quarter'] = sales_df['month'].apply(get_quarter)

# Creazione della variabile 'week'
sales_df['week'] = pd.to_datetime(sales_df['time_code'], format='%Y%m%d').dt.isocalendar().week

# Creazione della variabile 'day_of_week'
sales_df['day_of_week'] = pd.to_datetime(sales_df['time_code'], format='%Y%m%d').dt.day_name()

dict_colonne = dict_from_header(df.columns.tolist(), df.values.tolist())
gen table distinct(dict_colonne, colonne da aggiungere)
set primary key(gen table distinct, nome_id)

# In[12]:


dict_sales=dict_from_header(sales_df.columns.tolist(), sales_df.values.tolist())


# In[13]:


time_table=gen_table_distinct(dict_sales, 'day', 'month', 'year', 'week', 'quarter', 'day_of_week')
time_table=set_primary_key(time_table, 'time_id')


# In[14]:


time_table


# In[15]:


time_df=tuples_to_dataframe(time_table)
time_df


# # CREAZIONE TABELLA CPU

# In[16]:


cpu_table=gen_table_distinct(dict_sales, 'cpu_vendor_name', 'cpu_brand', 'cpu_series', 'cpu_name', 'cpu_n_cores', 'cpu_socket')
cpu_table=set_primary_key(cpu_table, 'cpu_id')
pd.DataFrame(cpu_table)


# In[17]:


cpu_df=tuples_to_dataframe(cpu_table)
cpu_df


# # CREAZIONE TABELLA RAM

# In[18]:


pd.DataFrame(cpu_table).isnull().sum()


# In[19]:


ram_table=gen_table_distinct(dict_sales, 'ram_vendor_name', 'ram_brand', 'ram_size', 'ram_type', 'ram_clock')
ram_table=set_primary_key(ram_table, 'ram_id')
ram_df=tuples_to_dataframe(ram_table)
ram_df


# In[20]:


ram_df.duplicated().sum()


# In[21]:


ram_df[3680:3710]


# # CREAZIONE TABELLA GPU

# In[22]:


gpu_table=gen_table_distinct(dict_sales, 'gpu_vendor_name', 'gpu_brand', 'gpu_processor_manufacturer', 'gpu_memory', 'gpu_memory_type')
gpu_table=set_primary_key(gpu_table, 'gpu_id')


# In[23]:


gpu_df=tuples_to_dataframe(gpu_table)
gpu_df


# # CREAZIONE TABELLA GEOGRAPHY

# In[24]:


geo_df = pd.read_csv("C:\\Users\\HP\\Downloads\\Archivio\\geography.csv")
geo_df.shape


# In[25]:


geo_df.isnull().sum()


# In[26]:


geo_df["geo_id"].dtype


# In[27]:


geo_df['geo_id'] = geo_df['geo_id'].astype(str)


# In[29]:


sales_df['geo_id'] = sales_df['geo_id'].astype(str)


# In[30]:


merged_df=pd.merge(geo_df, sales_df, on="geo_id", how="outer")
merged_df.shape


# In[34]:


merged_df.isnull().sum()


# In[35]:


dict_merged=dict_from_header(merged_df.columns.tolist(), merged_df.values.tolist())


# In[36]:


geography_table=gen_table_distinct(dict_merged, 'country', 'region', 'continent', 'currency')
geography_table=set_primary_key(geography_table, 'geo_id')


# In[37]:


geography_df=tuples_to_dataframe(geography_table)
geography_df


# # CREAZIONE TABELLA COMPUTER SALES

# In[38]:


merged_df["currency"].unique()


# In[39]:


# Definisci i coefficienti di conversione per ogni valuta rispetto all'USD
conversion_rates = {'EUR': 1.2, 'AUD': 0.7, 'GBP': 1.4, 'CAD': 0.8, 'NZD': 0.65, 'USD': 1.0}

merged_df['ram_sales']=merged_df['ram_sales_currency']
merged_df['cpu_sales']=merged_df['cpu_sales_currency']
merged_df['gpu_sales']=merged_df['gpu_sales_currency']
merged_df['total_sales']=merged_df['sales_currency']


# Calcola i valori *_usd utilizzando i coefficienti di conversione
merged_df['ram_sales_usd'] = merged_df['ram_sales'] * merged_df['currency'].replace(conversion_rates).round(2)
merged_df['cpu_sales_usd'] = merged_df['cpu_sales'] * merged_df['currency'].replace(conversion_rates).round(2)
merged_df['gpu_sales_usd'] = merged_df['gpu_sales'] * merged_df['currency'].replace(conversion_rates).round(2)
merged_df['total_sales_usd'] = merged_df['total_sales'] * merged_df['currency'].replace(conversion_rates).round(2)


# In[40]:


import pandas as pd

def add_foreign_key(df1, df2, foreign_key_name):
    """
    Aggiunge una chiave esterna (df2_id) a df1 basandosi sui valori corrispondenti nelle colonne comuni tra df1 e df2.

    Args:
    df1 (DataFrame): Il DataFrame principale.
    df2 (DataFrame): Il DataFrame che contiene la chiave primaria e le colonne per il join.
    foreign_key_name (str): Il nome della chiave primaria in time_df.

    Returns:
    DataFrame: Il DataFrame sales_df con la colonna aggiunta.
    """
    # Trova l'intersezione delle colonne in comune escludendo foreign_key_name
    common_cols = list(set(df1.columns) & set(df2.columns))

    # Esegui un'operazione di join basata sulle colonne comuni
    merged_df = pd.merge(df1, df2, on=common_cols)

    # Aggiungi la colonna foreign_key_name a sales_df basandoti sui valori delle colonne comuni
    df1[foreign_key_name] = merged_df[foreign_key_name]

    return df1


# In[41]:


merged_df = add_foreign_key(merged_df, gpu_df, 'gpu_id')
merged_df = add_foreign_key(merged_df, cpu_df, 'cpu_id')
merged_df = add_foreign_key(merged_df, ram_df, 'ram_id')
merged_df = add_foreign_key(merged_df, time_df, 'time_id')


# In[57]:


geography_df.columns


# In[58]:


merged_df.columns


# In[59]:


merged_df = merged_df.drop(columns=['geo_id'])


# In[60]:


merged_df = add_foreign_key(merged_df, geography_df, 'geo_id')


# In[61]:


merged_df.head(5)


# In[62]:


dict_merged=dict_from_header(merged_df.columns.tolist(), merged_df.values.tolist())


# In[63]:


computer_sales_table=gen_table_distinct(dict_merged, 'geo_id', 'time_id', 'ram_id', 'cpu_id', 'gpu_id', 'ram_sales', 
                                        'ram_sales_usd', 'cpu_sales', 'cpu_sales_usd', 
                                        'gpu_sales', 'gpu_sales_usd', 'total_sales', 'total_sales_usd')
computer_sales_table=set_primary_key(computer_sales_table, 'sale_id')


# In[64]:


computer_sales_table


# In[65]:


computer_sales_df=tuples_to_dataframe(computer_sales_table)
computer_sales_df


# In[66]:


computer_sales_df.dtypes


# In[67]:


time_df.dtypes


# In[68]:


ram_df.dtypes


# In[69]:


cpu_df.dtypes


# In[70]:


gpu_df.dtypes


# In[71]:


geography_df.dtypes


# In[ ]:





# # CARICAMENTO NEL SERVER

# In[72]:


import os

tables = dict()
table_names = ['TIME', 'RAM', 'GPU', 'GEOGRAPHY', 'COMPUTER_SALES',  'CPU']

# creating a copy of each to run this cell multiple times
tables_saved = [time_table[:], ram_table[:], gpu_table[:], geography_table[:], computer_sales_table[:], cpu_table[:]]

for table_name, table in zip(table_names, tables_saved):
    header = table.pop(0) # getting the header from each table
    tables[table_name] = dict_from_header(header, table) #storing each table to a dictionary called tables ()V


# In[73]:


from tqdm import tqdm
import pyodbc
import copy
import re

import os

tables = dict()
table_names = ['TIME', 'RAM', 'CPU', 'GPU', 'GEOGRAPHY', 'COMPUTER_SALES']

# creating a copy of each to run this cell multiple times
tables_saved = [time_table[:], ram_table[:], cpu_table[:], gpu_table[:], geography_table[:], computer_sales_table[:]]

for table_name, table in zip(table_names, tables_saved):
    header = table.pop(0) # getting the header from each table
    tables[table_name] = dict_from_header(header, table) #storing each table to a dictionary called tables ()V

#this class will empty the remote table to load the one from a dictionary

class Upload_Table():
    
    def __init__(self, table_dict, table_name):
        #to avoid editing the original dictionary (if error occurs)
        self.table = copy.deepcopy(table_dict)
        #it removes ambiguities in the case of reserved keywords (e.g. User)
        self.table_name = "["+table_name+"]" 
        
        #Create a connection and a cursor in the database
        self.conn = self.get_connection()
        self.cursor = self.conn.cursor()
        
        #adjust the table input in the class to the types in the SQL Server Schema
        self.table = self.adjust_types()
        
        #try to upload the table
        try:
            self.insert_into_table()
            
        #close connection if an exception occurs
        except Exception as e:
            self.cursor.close()
            self.conn.close()
            raise e
            
        #close connection if it is a success
        self.cursor.close()
        self.conn.close()
        
        #delete the connection variables from the class
        del self.cursor
        del self.conn
        
    #function to get the credentials and perform the connection to the database
    def get_connection(self):
        
        #a file with the ip, userid and credentials must be in the same folder
          
        driver = 'ODBC Driver 17 for SQL Server'
        self.db = 'Group_ID_778_DB' #the name of the database to which I am operating
        ip = 'lds.di.unipi.it'
        uid = 'Group_ID_778'
        pwd = 'BTUUP482'

        conn = pyodbc.connect(f'DRIVER={driver};SERVER=tcp:{ip};DATABASE={self.db};UID={uid};PWD={pwd}')
        
        return conn
    
    def adjust_types(self):
        self.cursor.execute(f'SELECT * FROM {self.table_name}')
        
        #using a dictionary to cast the correct types to the data
        #the lambda functions is there to cast the correct types
        cast_types = {'int': lambda x: int(float(x)), #some strings have values with a dot
                      'float': float, 
                      'str': lambda x: f"'{str(x)}'", #string
                      'datetime.date': lambda x: f"'{str(x)}'",  #date must be passed as a string in explicit queries
                      'bool': lambda x: int(float(x))}
        
        
        col_type = dict()

        #looping across the information get by the cursor
        for name_col, type_col, _, len_char1, len_char2, _, accept_none in self.cursor.description:
            #getting the type from the type_col response string
            str_type = re.findall("'.*'", str(type_col))[0].strip("'")
            #saving the column with the corresponding type to cast into a dictionary
            col_type[name_col] = cast_types[str_type]

        #get the header of the local table
        self.header_table = list(self.table.keys())
        #check if the header of the local table corresponds to the header in the server
        assert list(col_type.keys()) == self.header_table, f'The header ({self.header_table}) of the table and the table in the Server ({list(col_type.keys())}) do not match!'

        
        table_list = list()

        #cast the correct types to the local table
        for col in self.header_table:
            to_type = col_type[col] #get the stored type recast function from the col_type dictionary
            self.table[col] = [to_type(el) for el in self.table[col]] #recast each element of the column
            table_list.append([col] + self.table[col]) #save a copy and add the header to the column
            
        table_list = list(zip(*table_list)) #rebuild the table from the recasted columns (list of lists)
        
        return table_list

    def sql_query_maker(self):
        #add the first part of the query with table name and the rest
        sql_query = f"INSERT INTO {self.table_name} ({', '.join(self.header_table)}) VALUES ("
        first_parameter = '{}'
        sql_query += first_parameter
        
        #for each element in the header add the parametric question mark (except for the first, thus -1)
        for i in range(len(self.header_table)-1):
            sql_query += ", {}"
        sql_query += ")" #close the row to upload

        return sql_query

    def delete_previous_vals_from_table(self, table_name):
        #try to delete the values from the table considered to upload
        try:
            self.cursor.execute(f'DELETE FROM {table_name}')
            
        #Every data in the hierarchy of the table will be deleted to avoid the Integrity Error
        except pyodbc.IntegrityError as ierr:
            #looking for the table to DELETE FROM in the error with regex
            table_prefix = self.db[:self.db.rfind('_')]
            start_idx = re.search(f'The conflict occurred in database "{self.db}", table "{table_prefix}\.', str(ierr)).end()
            end_idx = str(ierr)[start_idx:].find('"')+start_idx
            
            new_table = str(ierr)[start_idx:end_idx]
            new_table_name = "["+new_table+"]"
            
            #recursively remove from the tables in the higher hierarchy
            self.delete_previous_vals_from_table(new_table_name)
            #retry removing from the table (it should work now)
            self.cursor.execute(f'DELETE FROM {table_name}')
            
    def insert_into_table(self):
        #getting the query
        model_sql_query = self.sql_query_maker()
        #removing all the values from table to upload it
        self.delete_previous_vals_from_table(self.table_name)
        
        print("Query:\n" + model_sql_query.format(*'?'*len(self.header_table)))
        
        sql_query = ''

        #tqdm gives the progress bar, I looped across the rows (avoiding the header)
        for n, row in enumerate(tqdm(self.table[1:], ascii=True, desc='Uploading Progress')):
            tupla = tuple(el for el in row) #making the row a tuple if it is not
            
            current_query = model_sql_query.format(*tupla)+';\n' # inserting values into the query
            sql_query += current_query # adding up the current query to the others up until 100 queries
                        
            # To commit every 100 records (in case it crashes and I avoid the delete statement to finish uploading later)
            # It avoids to commit at the first row but it commits after the last
            # (len-2 because one is the header, the other is the index of the last element in a zero indexing base)
            if (n == (n // 100) * 100) and n != 0 or n == len(self.table) - 2:
                
                # Try to reconnect at least 10 times if the execution fails
                for attempt in range(10):
                    try:
                        #executing the 100 queries
                        self.cursor.execute(sql_query)
                        break

                    #if it reaches the 10th execution raise the error I blocked
                    except Exception as e:
                        if attempt == 9:
                            print(sql_query)
                            raise e
                        else:
                            continue

                self.conn.commit() #either way commit everything
                sql_query = '' #reset the query so that it a new group of queries can be committed
                
        #commit at the end
        self.conn.commit()


# In[74]:


# upload all tables in the dictionary created (table names and tables were the input in the dict creation)
for table_name in table_names:
    try:
        Upload_Table(tables[table_name], table_name)
    except Exception as e:
        print(f"Errore durante il caricamento della tabella {table_name}: {e}")


# In[75]:



end_time = datetime.now() # to measure total execution time
print(f'Duration: {(end_time - start_time)}') 


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




