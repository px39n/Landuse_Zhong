# Landuse_Zhong


## 环境配置

### GPU计算环境
```

wsl
source ~/bayes-gpu/bin/activate
jupyter notebook --no-browser --port=8888

```
然后在本地浏览器打开 http://localhost:8888  或者 jupyter notebook --no-browser --port=8888 并输入上面显示的token，连接服务器。



在Jupyter页面中，选择 `Python (bayes-gpu)` 作为核（kernel），以确保在GPU加速环境下运行代码。