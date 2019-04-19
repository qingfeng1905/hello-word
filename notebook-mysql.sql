/* *******************************************************************
Mysql学习笔记
@brief: explore mysql
@brief: Luffy
********************************************************************/


-- 安装mysql终端：64位
	-- MySQL Community Edition
	-- MySQL Installer 5.7.21

		
-- 移除/安装mysql服务器
	-- 移除服务：
		-- 进入dos系统（Win+R, cmd），输入命令：sc delete MySQL57
	-- 安装服务：
		-- 在dos系统中，进入mysql的bin目录下，本机中为：C:\Program Files\MySQL\MySQL Server 5.7\bin
		-- 执行命令：mysqld.exe --install MySQL57 --defalts-file="my.ini文件路径"
		-- 本机中，my.ini文件路径为：C:\ProgramData\MySQL\MySQL Server 5.7\my.ini	
	-- 重命名：先删除，在安装
	-- 若服务器关闭，则通过mysql客户端无法登陆，Python也无法连接；
	

-- 数据库迁移，How？
	-- 数据所在物理位置：C:\ProgramData\MySQL\MySQL Server 5.7\Data

	
	
-- 数据库用户管理：创建、查看、授权、删除：https://blog.csdn.net/u014453898/article/details/55064312
-- 权限管理： https://www.cnblogs.com/SQL888/p/5748824.html
select user,host from mysql.user;		-- 查询用户
create user luffy@localhost identified by '123';		-- 创建用户
drop user luffy@localhost;


-- 常用命令：
show databases;		-- 数据库列表

create database testdb;		-- 创建数据库
use testdb;		-- 使用该数据库
show tables;	-- 该数据库下的表
desc table_name; 	-- 表字段信息
describe table_name;	-- 表字段信息

select * from table_name;		-- 查询表信息

	
-- 使用python连接mysql
/* 安装MySQLdb
Ref: https://www.cnblogs.com/swje/p/7979089.html

# 在spyder中操作比上述简单
!pip list  #已安装wheel
!pip install wheel

# 将下载的文件放在当前工作目录下，然后直接用pip安装，命令如下：
!pip install mysqlclient-1.3.12-cp36-cp36m-win_amd64.whl

Python使用MySQL数据库的方法以及一个实例：
https://www.cnblogs.com/itdyb/p/5700614.html
*/


import MySQLdb as mysql
# 打开数据库连接
db = mysql.connect("localhost","root","891124","testdb" )
# 使用cursor()方法获取操作游标 
cursor = db.cursor()
# 使用execute方法执行SQL语句
cursor.execute("SELECT VERSION()")
# 使用 fetchone() 方法获取一条数据
data = cursor.fetchone()
print ("Database version : %s " % data)


""" 创建数据库表：employee """
# 若数据表已存在则删除
cursor.execute("drop table if exists employee")
# 创建数据表sql语句
sql = """CREATE TABLE EMPLOYEE (
         FIRST_NAME  CHAR(20) NOT NULL,
         LAST_NAME  CHAR(20),
         AGE INT,  
         SEX CHAR(1),
         INCOME FLOAT )"""

cursor.execute(sql)


""" 数据库插入操作 """
# SQL 插入语句
sql = """INSERT INTO EMPLOYEE(FIRST_NAME,
         LAST_NAME, AGE, SEX, INCOME)
         VALUES ('Tong', 'Wu', 15, 'F', 2000)"""
try:
   # 执行sql语句
   cursor.execute(sql)
   # 提交到数据库执行
   db.commit()
except:
   # Rollback in case there is any error
   db.rollback()

   
""" 数据库查询操作 """
sql = "select * from employee where SEX='F'"
cursor.execute(sql)
data = cursor.fetchall()
#data = cursor.fetchone()


""" 数据库更新操作 """
cursor.execute("update employee set INCOME = INCOME + 500 where SEX = 'F'")
db.commit()


""" 数据库删除操作 """
cursor.execute("delete from employee where FIRST_NAME = 'Mac'")
db.commit()



""" 关闭数据库连接 """
db.close()
