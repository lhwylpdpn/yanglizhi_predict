#!/usr/bin/python
# -*- coding: UTF-8 -*-


import time
import datetime
import sys
import csv
import json
import pandas as pd
import numpy as np
import requests
import os
from dateutil.relativedelta import relativedelta
from Config import Config
import cx_Oracle
from sqlalchemy import create_engine
import multiprocessing as mp

path = os.path.split(os.path.realpath(__file__))[0]
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)
# 设置value的显示长度为100，默认为50
pd.set_option('max_colwidth', 100)


class SAP_clac:

    def __init__(self):
        self.path = path + '/'
        self.df = pd.DataFrame()
        self.df_list = []
        # self.columnslist=['algorithm','start_year','end_year','month','WERKS','MATNR','algorithm_param','money','num']

    def get_data_resource(self):

        # 开始年和结束年本质没意义？后面的年月是实际值,年值2020 ，季值202001~04 ，月值202001~12
        # 工厂、物料号 和 金额、数量，实际上是要笛卡尔积乘开预测？也就是，假设2个工厂 2种物料，要预测4条数据出来
        # 工厂1物料1（金额、数量） 工厂1物料2（金额、数量） 工厂2物料1（金额、数量） 工厂2物料2（金额、数量）

        self.df = pd.read_csv(self.path + "月度.csv", sep="|")
        # METHOD算法|开始年|结束年|ZNY年季|WERKS工厂|MATNR物料号|VALUE变量a|ZJE总金额|SL数量|ZSFYC是否预测值
        self.df.columns = self.columnslist
        self.df['YQM_tag'] = 1  # 年月日标记 1为月,3 为季,12为年

    def clear_data(self):
        _type = self.df.drop_duplicates(subset=['ZLIFNR', 'MATNR', 'algorithm'], keep='first')
        _type = _type[['ZLIFNR', 'MATNR', 'algorithm']]

        for r in range(0, _type.shape[0]):
            temp = self.df[
                (self.df['ZLIFNR'] == _type.iloc[r]['ZLIFNR']) & (self.df['MATNR'] == _type.iloc[r]['MATNR']) & (
                        self.df['algorithm'] == _type.iloc[r]['algorithm'])]
            self.df_list.append(pd.DataFrame(temp.sort_values(by='month', ascending=True)))

    def get_df_list(self):
        return self.df_list



    def get_result(self):
        return  self.df_pred

    def clac_loss(self):
        pass

    def get_data_resource_from_oracle(self,table_name):
        connect_str=Config.oracle_param['user']+"/"+Config.oracle_param['password']+"@"+Config.oracle_param['host']+":"+str(Config.oracle_param['port'])+"/"+Config.oracle_param['schema']
        db = cx_Oracle.connect(connect_str)
        cursor= db.cursor()
        table_name_1=table_name
        #todo 上线前需要修改取sign 和最后两列的取值
        sql1="select CALYEAR,66,mandt,TYPE,case when UPPER(sign)='M' then 1  when UPPER(sign)='S' then 3 when UPPER(sign)='Y' then 12 else 0 end,zkey,year_beg,year_end,zny,werks,matnr,value,zje,sl,zsfyc,text,cpxh,cpzl,hbm,unit,zq,hxm,dj,pjjg,pjghzq,zlifnr,division  from  "+table_name_1
        # 66 用来占位而已 留出来算法种类的标记位


        self.df=pd.read_sql(sql1,db)

        cursor.close()
        db.close()
        #print(self.df.shape)
        self.columnslist=['CALYEAR','algorithm','MANDT','TYPE','YQM_tag','ZKEY','YEAR_BEG','YEAR_END','month','WERKS','MATNR','algorithm_param','money','num','ZSFYC','TEXT','CPXH','CPZL','HBM','UNIT','ZQ','HXM','DJ','PJJG','PJGHZQ', 'ZLIFNR', 'DIVISION']
        #print(len(self.columnslist))
        self.df_pred=pd.DataFrame(columns=self.columnslist+['money_predict_loss','money_num_loss','loss_max'])
        self.df.columns=self.columnslist


    def clear_data_from_oracle(self):

        self.df_list = [group[1].sort_values(by='month', ascending=True).reset_index(drop=True) for group in
                        self.df.groupby(['ZLIFNR', 'MATNR', 'TYPE', 'YQM_tag', 'ZKEY', 'YEAR_BEG', 'YEAR_END'])]


    def get_columns(self):
        return self.columnslist
def main():


        x=time.time()
        obj_ = SAP_clac() # init 对象
        obj_.get_data_resource_from_oracle('ZYC_HISTORY') #从表取数据
        print('get_data_resource_from_oracle',time.time()-x)
        x = time.time()
        obj_.clear_data_from_oracle()
        print('clear_data_from_oracle', time.time() - x)
        x = time.time()
        res=obj_.get_df_list()
        print('get_df_list', time.time() - x)
        k=len(res)
        print('共有',k,'个待预测内容')


        x = time.time()
        table_name='ZYC_RESULT'
        n_processes = mp.cpu_count()
        pool = mp.Pool(processes=n_processes)
        columns = obj_.get_columns()
        args = [(columns,r,table_name) for r in res]  # 任务列表
        pool.starmap(parallel_func, args)  # 分配任务
        pool.close()
        pool.join()
        w = time.time() - x
        print('共进行了', k, '个内容预测', '共耗时', w, '秒', '平均耗时', w / k, '秒')

        obj_ = SAP_clac() # init 对象
        obj_.get_data_resource_from_oracle('ZYC_HISTORY_CB') #从表取数据
        obj_.clear_data_from_oracle() #转化格式
        res=obj_.get_df_list()
        k=len(res)
        print('共有',k,'个待预测内容')

        table_name='ZYC_RESULT_CB'
        x = time.time()
        n_processes = mp.cpu_count()
        pool = mp.Pool(processes=n_processes)
        columns = obj_.get_columns()
        args = [(columns, r,table_name) for r in res]  # 任务列表
        pool.starmap(parallel_func, args)  # 分配任务
        pool.close()
        pool.join()
        w = time.time() - x
        print('共进行了', k, '个内容预测', '共耗时', w, '秒', '平均耗时', w / k, '秒')


def algorithm_1(columnslist, r):  # 一次指数平滑

    r.index = range(0, len(r))  # 按照行号重置DF
    # if r.loc[0,'algorithm']=='1-一次平滑': #过滤算法1 去掉这类字段，所有内容都预测两次
    r['algorithm'] = 'Y'
    df = pd.DataFrame(r)
    df.loc[len(df)] = df.loc[len(df) - 1].to_dict()  # 增加出来一行值用来表达预测值
    # 日期转化，将上一行的日期转成py格式，再往后计算一个
    df.loc[len(df) - 1, 'month'] = datetime.datetime.strftime(
        datetime.datetime.strptime(str(df.loc[len(df) - 2, 'month']), '%Y%m').date() + relativedelta(
            months=+df.loc[len(df) - 2, 'YQM_tag']), '%Y%m')

    # 预测num

    df['y_pred_num_temp'] = df['algorithm_param'] * df['num']
    #print(df)
    #如果df 多于3行执行下面的
    #print(df.shape[0])
    if df.shape[0] >= 3:
        df.loc[0, 'y_pred_num_temp'] = (df.loc[0, 'num'] + df.loc[1, 'num'] + df.loc[2, 'num']) / 3  # 初始值预测值是最开始3个值的平均值
    else:
        df.loc[0, 'y_pred_num_temp'] = df.loc[0, 'num']
    df['y_pred_num'] = df['algorithm_param'] * df.shift(periods=1)['num'] + (1 - df['algorithm_param']) * \
                       df.shift(periods=1)['y_pred_num_temp']

    df.loc[len(df) - 1, 'num'] = df.loc[len(df) - 1, 'y_pred_num']
    # 预测money
    df['y_pred_money_temp'] = df['algorithm_param'] * df['money']
    if df.shape[0] >= 3:
        df.loc[0, 'y_pred_money_temp'] = (df.loc[0, 'money'] + df.loc[1, 'money'] + df.loc[2, 'money']) / 3  # 初始值预测值是最开始3个值的平均值
    else:
        df.loc[0, 'y_pred_money_temp'] = df.loc[0, 'money']
    df['y_pred_money'] = df['algorithm_param'] * df.shift(periods=1)['money'] + (1 - df['algorithm_param']) * \
                         df.shift(periods=1)['y_pred_money_temp']
    ###20211123增加预测结果小数截取
    df["y_pred_money"] = df['y_pred_money'].apply(lambda x: round(float(x), 4))
    df["y_pred_num"] = df['y_pred_num'].apply(lambda x: round(float(x), 4))

    df.loc[len(df) - 1, 'money'] = df.loc[len(df) - 1, 'y_pred_money']
    df.loc[len(df) - 1, 'money_predict_loss'] = abs(
        df.loc[len(df) - 2, 'money'] - df.loc[len(df) - 2, 'y_pred_money'])
    df.loc[len(df) - 1, 'money_num_loss'] = abs(df.loc[len(df) - 2, 'num'] - df.loc[len(df) - 2, 'y_pred_num'])

    # 返回清理后的数据
    df = df[columnslist + ['money_predict_loss', 'money_num_loss']]
    df = df.iloc[-1:].reset_index(drop=True)
    return df
def algorithm_2_clac(x):
    lan = []
    for i in range(len(x)):
        if i == len(x) - 1:
            continue
        #lan.append(x[i] / x[i + 1])
        lan.append(x[i] / x[i + 1] if x[i + 1] != 0 else x[i] / 0.00001) # 增加0.000001的逻辑20211118 与业务核对确认，0-1 认为无穷大

    x_1 = np.cumsum(x)

    B = np.array([-1 / 2 * (x_1[i] + x_1[i + 1]) for i in range(len(x) - 1)])
    B = np.mat(np.vstack((B, np.ones((len(x) - 1,)))).T)
    Y = np.mat([x[i + 1] for i in range(len(x) - 1)]).T
    u = np.dot(np.dot(B.T.dot(B).I, B.T), Y)
    [a, b] = [u[0, 0], u[1, 0]]
    a_new, b = x[0] - b / a, b / a

    # 输入需要预测的年数
    year = 1
    year += len(x)
    x_predict = [x[0]]
    x_predict = x_predict + [a_new * (np.exp(-a * i) - np.exp(-a * (i - 1))) for i in range(1, year)]
    return x_predict
def algorithm_2(columnslist, r):  # Gm（1.1）

            r.index = range(0, len(r))#按照行号重置DF
        #if r.loc[0, 'algorithm'] == '2-灰度算法':  # 过滤算法1
            df = pd.DataFrame(r)
            r['algorithm']='H' # 人工打上算法标记
            x=df['num'].tolist() # num 转成要预测的序列
            y=df['money'].tolist()
            df.loc[len(df)] = df.loc[len(df) - 1].to_dict()# 增加出来一行值用来表达预测值
            df.loc[len(df) - 1, 'month'] = datetime.datetime.strftime(datetime.datetime.strptime(str(df.loc[len(df) - 2, 'month']), '%Y%m').date() + relativedelta(months=+df.loc[len(df) - 2, 'YQM_tag']), '%Y%m')
            #如果数据上本身存在问题，那么会产生奇异值矩阵，直接标记0，表示无法预测
            try:
                x_predict = self.algorithm_2_clac(x)
                y_predict = self.algorithm_2_clac(y)
            except:
                x_predict = x+[0]
                y_predict = y+[0]
            df['y_pred_num'] = x_predict
            df['y_pred_money'] = y_predict
            x_predict = [0 if v < 0 else v for v in x_predict]
            y_predict = [0 if v < 0 else v for v in y_predict]
            df["y_pred_money"] = df['y_pred_money'].apply(lambda x: round(float(x), 4))
            df["y_pred_num"] = df['y_pred_num'].apply(lambda x: round(float(x), 4))
            #print(x_predict)
            #print(y_predict)
            df.loc[len(df) - 1, 'num'] = df.loc[len(df) - 1, 'y_pred_num']
            df.loc[len(df) - 1, 'money'] = df.loc[len(df) - 1, 'y_pred_money']

            df.loc[len(df) - 1, 'money_predict_loss'] = abs(df.loc[len(df) - 2, 'money'] - df.loc[len(df) - 2, 'y_pred_money'])
            df.loc[len(df) - 1, 'money_num_loss'] = abs(df.loc[len(df) - 2, 'num'] - df.loc[len(df) - 2, 'y_pred_num'])

            # 返回清理后的数据
            df = df[columnslist + ['money_predict_loss', 'money_num_loss']]
            #只返回最后一行预测值,转成新的pd
            df = df.iloc[-1:].reset_index(drop=True)
            return df


def update_result_loss_max(dfA,dfB):#x=0 算法1大 x=1 算法2 大
    len_=len(dfA)
    #print(len_)
    #取dfa最后一行的money_num_loss
    #print(dfA,dfB)
    x=0 if dfA.loc[len_-1,'money_num_loss']>dfB.loc[len_-1,'money_num_loss'] else 1
    #A>B x=0 ;A<B x= 1
    dfA.loc[len_-1 ,'loss_max']=x
    dfB.loc[len_-1 ,'loss_max']=x+1 if x==0 else x-1
    return dfA,dfB



def insert_result_to_oracle(df_pred,table_name):


    # 连接数据库
    connect_str=Config.oracle_param['user']+"/"+Config.oracle_param['password']+"@"+Config.oracle_param['host']+":"+str(Config.oracle_param['port'])+"/"+Config.oracle_param['schema']
    db = cx_Oracle.connect(connect_str)
    cursor= db.cursor()

    list_temp=df_pred.columns.to_list()
    list_temp = ['METHOD' if i == 'algorithm' else i for i in list_temp]
    list_temp = ['SIGN' if i == 'YQM_tag' else i for i in list_temp]
    # list_temp.remove('ZKEY')
    list_temp = ['ZNY' if i == 'month' else i for i in list_temp]
    list_temp = ['VALUE' if i == 'algorithm_param' else i for i in list_temp]
    list_temp = ['ZJE' if i == 'money' else i for i in list_temp]
    list_temp = ['SL' if i == 'num' else i for i in list_temp]
    list_temp = ['ZY' if i == 'loss_max' else i for i in list_temp]



    list_temp = [s.lower() for s in list_temp]  # oracle  入库要求所有字段必须小写
    df_pred.columns = list_temp
    df_pred.drop('money_predict_loss', inplace=True, axis=1)
    df_pred.drop('money_num_loss' , inplace=True, axis=1)


    df_pred.fillna(value=0, inplace=True)  # 各种错误数据带来的计算中有None 的替换成0
    df_pred['sign'] = df_pred['sign'].apply(lambda x: str(x).replace('12', 'Y').replace('3', 'S').replace('1', 'M'))

    list_temp= df_pred.columns.to_list()
    list_temp.remove('zje')
    list_temp.remove('sl')
    list_temp.remove('zy')
    primary_key= list_temp


    for row in df_pred.itertuples(index=False):
        #将row转换成dict格式，key是index，value是value
        row_dict = row._asdict()

        v=",".join(["'"+str(v)+"' as "+str(k) for k,v in row_dict.items()])
        merge_sql = f"""
        MERGE INTO {table_name} dest
        USING (SELECT {v} FROM dual) src
        ON ({' and '.join(['dest.' + col + ' = src.' + col for col in primary_key])})
        WHEN MATCHED THEN
            UPDATE SET dest.zje = src.zje, dest.sl = src.sl, dest.zy = src.zy
        WHEN NOT MATCHED THEN
            INSERT ({', '.join(primary_key + ['zje', 'sl','zy'])})
            VALUES ({', '.join(['src.' + col for col in primary_key])}, src.zje, src.sl, src.zy)
        """
        cursor.execute(merge_sql)
        db.commit()
    cursor.close()
    db.close()


def parallel_func(columnslist,r,table_name):
    #print(r.shape)
    dfA=algorithm_1(columnslist,r)
    dfB=algorithm_2(columnslist,r)
    dfA,dfB=update_result_loss_max(dfA,dfB)
    #纵向合并dfA,dfB
    df_pred=pd.concat([dfA,dfB],axis=0)

    insert_result_to_oracle(df_pred=df_pred,table_name=table_name)
    return



#增加一个专门测试数据库的函数 #
def test(db_name):
    connect_str = Config.oracle_param['user'] + "/" + Config.oracle_param['password'] + "@" + Config.oracle_param[
        'host'] + ":" + str(Config.oracle_param['port']) + "/" + Config.oracle_param['schema']
    db = cx_Oracle.connect(connect_str)
    cursor = db.cursor()
    sql1 = "select count(*) from " + db_name
    df = pd.read_sql(sql1, db)
    print(db_name,df)
    cursor.close()
    db.close()
if __name__ == '__main__':
    main()
    test("ZYC_RESULT")
    test("ZYC_RESULT_CB")


