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
path=os.path.split(os.path.realpath(__file__))[0]
pd.set_option('display.max_columns', None)
#显示所有行
pd.set_option('display.max_rows', None)
#设置value的显示长度为100，默认为50
pd.set_option('max_colwidth',100)

class SAP_clac:

    def __init__(self):
        self.path=path+'/'
        self.df=pd.DataFrame()
        self.df_list=[]
        self.columnslist=['algorithm','start_year','end_year','month','WERKS','MATNR','algorithm_param','money','num']
        self.df_pred=pd.DataFrame(columns=self.columnslist+['money_predict_loss','money_num_loss'])
    def  get_data_resource(self):

        #开始年和结束年本质没意义？后面的年月是实际值,年值2020 ，季值202001~04 ，月值202001~12
        #工厂、物料号 和 金额、数量，实际上是要笛卡尔积乘开预测？也就是，假设2个工厂 2种物料，要预测4条数据出来
        #工厂1物料1（金额、数量） 工厂1物料2（金额、数量） 工厂2物料1（金额、数量） 工厂2物料2（金额、数量）

        self.df=pd.read_csv(self.path+"月度.csv", sep="|")
        #METHOD算法|开始年|结束年|ZNY年季|WERKS工厂|MATNR物料号|VALUE变量a|ZJE总金额|SL数量|ZSFYC是否预测值
        self.df.columns=self.columnslist
        self.df['YQM_tag']=1 #年月日标记 1为月,3 为季,12为年


    def clear_data(self):
        _type=self.df.drop_duplicates(subset=['WERKS','MATNR','algorithm'],keep='first')
        _type=_type[['WERKS','MATNR','algorithm']]

        for r in range(0,_type.shape[0]):
            temp = self.df[
                (self.df['WERKS'] == _type.iloc[r]['WERKS']) & (self.df['MATNR'] == _type.iloc[r]['MATNR']) & (
                        self.df['algorithm'] == _type.iloc[r]['algorithm'])]
            self.df_list.append(pd.DataFrame(temp.sort_values(by = 'month',ascending=True)))

    def algorithm_1(self):#一次指数平滑
        for r in self.df_list:
            r.index = range(0, len(r))#按照行号重置DF
            if r.loc[0,'algorithm']=='1-一次平滑': #过滤算法1
                df=pd.DataFrame(r)
                df.loc[len(df)] =df.loc[len(df)-1].to_dict() #增加出来一行值用来表达预测值
                #日期转化，将上一行的日期转成py格式，再往后计算一个
                df.loc[len(df) - 1,'month']=datetime.datetime.strftime(datetime.datetime.strptime(str(df.loc[len(df) - 2,'month']), '%Y%m').date()+relativedelta(months=+df.loc[len(df) - 2,'YQM_tag']),'%Y%m')

                #预测num

                df['y_pred_num_temp']=df['algorithm_param']*df['num']
                df.loc[0,'y_pred_num_temp']=(df.loc[0,'num']+df.loc[1,'num']+df.loc[2,'num'])/3 # 初始值预测值是最开始3个值的平均值
                df['y_pred_num']=df['algorithm_param']*df.shift(periods=1)['num']+(1-df['algorithm_param'])*df.shift(periods=1)['y_pred_num_temp']
                df.loc[len(df) - 1,'num']= df.loc[len(df) - 1,'y_pred_num']
                #预测money
                df['y_pred_money_temp']=df['algorithm_param']*df['money']
                df.loc[0,'y_pred_money_temp']=(df.loc[0,'money']+df.loc[1,'money']+df.loc[2,'money'])/3 # 初始值预测值是最开始3个值的平均值
                df['y_pred_money']=df['algorithm_param']*df.shift(periods=1)['money']+(1-df['algorithm_param'])*df.shift(periods=1)['y_pred_money_temp']
                df.loc[len(df) - 1,'money']= df.loc[len(df) - 1,'y_pred_money']
                df.loc[len(df) - 1,'money_predict_loss']=abs(df.loc[len(df) - 2,'money']-df.loc[len(df) - 2,'y_pred_money'])
                df.loc[len(df) - 1, 'money_num_loss'] = abs(df.loc[len(df) - 2, 'num'] - df.loc[len(df) - 2, 'y_pred_num'])

                #返回清理后的数据
                df=df[self.columnslist+['money_predict_loss','money_num_loss']]
                tmp_dict=df.loc[len(df)-1].to_dict()
                self.df_pred.loc[len(self.df_pred)] =  tmp_dict

    def algorithm_2_clac(self,x):
        lan = []
        for i in range(len(x)):
            if i == len(x) - 1:
                continue
            lan.append(x[i] / x[i + 1])
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

    def algorithm_2(self):#Gm（1.1）
        for r in self.df_list:
            r.index = range(0, len(r))#按照行号重置DF
            if r.loc[0, 'algorithm'] == '2-灰度算法':  # 过滤算法1
                df = pd.DataFrame(r)
                x=df['num'].tolist() # num 转成要预测的序列
                y=df['money'].tolist()
                df.loc[len(df)] = df.loc[len(df) - 1].to_dict()# 增加出来一行值用来表达预测值
                df.loc[len(df) - 1, 'month'] = datetime.datetime.strftime(datetime.datetime.strptime(str(df.loc[len(df) - 2, 'month']), '%Y%m').date() + relativedelta(months=+df.loc[len(df) - 2, 'YQM_tag']), '%Y%m')

                x_predict = self.algorithm_2_clac(x)
                y_predict = self.algorithm_2_clac(y)
                df['y_pred_num']=x_predict
                df['y_pred_money'] = y_predict

                # MSE
                #print((np.array(x_predict[:len(x)]) - np.array(x[:len(x)])) / np.array(x[:len(x)]))

                df.loc[len(df) - 1, 'num'] = df.loc[len(df) - 1, 'y_pred_num']
                df.loc[len(df) - 1, 'money'] = df.loc[len(df) - 1, 'y_pred_money']
                print(df)
                df.loc[len(df) - 1, 'money_predict_loss'] = abs(df.loc[len(df) - 2, 'money'] - df.loc[len(df) - 2, 'y_pred_money'])
                df.loc[len(df) - 1, 'money_num_loss'] = abs(df.loc[len(df) - 2, 'num'] - df.loc[len(df) - 2, 'y_pred_num'])

                # 返回清理后的数据
                df = df[self.columnslist + ['money_predict_loss', 'money_num_loss']]
                tmp_dict = df.loc[len(df) - 1].to_dict()
                self.df_pred.loc[len(self.df_pred)] = tmp_dict



    def get_result(self):
        return  self.df_pred


    def test():
        db = cx_Oracle.connect('ITL01/1qazxsw@10.9.5.228:1521/IDS')
        print(db.version)
        cursor= db.cursor()
        table_name_1='sapsr3.ZYC_HISTORY'
        table_name_2='sapsr3.ZYC_HISTORY_CB'
        sql1='select TYPE,sign,zkey,year_beg,year_end,zny,werks,matnr,value,zje,sl,zsfyc,text,cpxh,cpzl,hbm,unit,zq,hxm,dj from  '+table_name_1
        sql2='select TYPE,sign,zkey,year_beg,year_end,zny,werks,matnr,value,zje,sl,zsfyc,text,cpxh,cpzl,hbm,unit,zq,hxm,dj from  '+table_name_2
        table_result_name_1=''
        table_result_name_2=''



        db.execute(sql)


if __name__ == '__main__':
    obj_=SAP_clac()
    obj_.get_data_resource()
    obj_.clear_data()
    obj_.algorithm_1()
    obj_.algorithm_2()
    print(obj_.get_result())