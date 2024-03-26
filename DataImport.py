#-*- coding: utf-8 -*-

import pymysql.cursors
import pandas as pd

#直接把数据库参数加到connect的语句里：
connector = pymysql.connect(host='xx.x.xx.xxx',port = 3306,user = 'root',password = 'xxx',  db="xxx", charset="utf8")

#创建游标
cursor = connector.cursor()

# 查询表格的所需字段数据
cursor.execute("SELECT * FROM hp_cylinder_dcs_data_model")

#获取 t_test所有数据
result = cursor.fetchmany(1000)
#print(result)

df_result = pd.DataFrame(list(result),columns = ["user_id", "file_id"])

print(df_result.shape)
print (df_result.head())

#sql语句
#sqlcmd="select col_name,col_type,col_desc from itf_datadic_dtl_d limit 10"
 
#利用pandas 模块导入mysql数据
#a=pd.read_sql(sqlcmd,dbconn)

# 关闭游标对象
cursor.close()
# 关闭连对象
connector.close()

