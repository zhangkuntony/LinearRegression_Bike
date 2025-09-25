import pandas as pd
import matplotlib.pyplot as plt
from holoviews.plotting.bokeh.styles import marker
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn import linear_model


def data_processor():
    # 读取数据
    data_set = pd.read_csv('./data/bike-day.csv')

    # 去除casual和registered特征，因为这些特征无法帮我们从单个用户的角度来模拟需求。casual + registered = cnt
    data_set = data_set.drop(['casual', 'registered'], axis=1)

    print(data_set.head())
    print(data_set.info())

    # instant和dteday不作为特征，使用drop()函数去除
    data_set = data_set.drop(['instant', 'dteday'], axis=1)

    # 用info()函数查看数据集是否去除instant和dteday特征
    print('用info()函数查看数据集是否去除instant和dteday特征')
    print(data_set.info())

    # 使用df.columns.values查看表头
    print('使用df.columns.values查看表头')
    print(data_set.columns.values)

    # cnt为标签列，其他均为特征，建立一个features的列表，去除cnt
    print('cnt为标签列，其他均为特征，建立一个features的列表，去除cnt')
    feature_list = data_set.columns.to_list()
    feature_list.remove('cnt')
    print(feature_list)
    return feature_list, data_set

def model_train_predict(x_train, x_test, y_train, y_test):
    # 初始化模型
    model_lr = linear_model.LinearRegression()

    # 训练模型
    model_lr.fit(x, y)

    # 模型预测
    predictions = model_lr.predict(x_test)
    print('predictions.shape:', predictions.shape)
    return predictions

def draw_linear_regression(y_test_data, prediction):
    print('绘制预测值与实际值的对比曲线')
    plt.style.use('fivethirtyeight')
    plt.figure(figsize=(16, 6))

    # 绘制测试集中的真实值
    plt.plot(y_test_data.values, marker='.', label='actual')

    # 绘制预测值
    plt.plot(prediction.flatten(), marker='.', label='predicted', color='red')

    # 图例位置
    plt.legend(loc='best')
    plt.show()

def calculate_mae_mse(y_test_data, prediction):
    mae_lr = mean_absolute_error(y_test_data, prediction)
    mse_lr = mean_squared_error(y_test_data, prediction)
    print('MAE_LR: {0}, MSE_LR: {1}'.format(mae_lr, mse_lr))

if __name__ == '__main__':
    # 获取处理之后的数据集和特征值
    features, dataset = data_processor()

    # 将数据集拆分为两个随机数据集，统一设置random_state种子参数为42
    # 设置test_size参数为0.3, 即将数据集中70%的数据分配给了训练集，而剩余30%的数据分配给测试集。
    x, x_test, y, y_test = train_test_split(dataset[features], dataset[['cnt']], test_size=0.3, random_state=42)
    print('x(训练集特征) shape is: {}'.format(x.shape))
    print('y(训练集标签) shape is: {}'.format(y.shape))
    print('-'*30)
    print('x(测试集特征) shape is: {}'.format(x_test.shape))
    print('y(测试集标签) shape is: {}'.format(y_test.shape))

    prediction_data = model_train_predict(x, x_test, y, y_test)

    # 绘制预测值与测试集中的真实值的对比曲线
    draw_linear_regression(y_test, prediction_data)

    # 评估模型计算精度
    calculate_mae_mse(y_test, prediction_data)