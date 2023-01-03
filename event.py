from flask import Flask, render_template, request
import joblib
import pickle
from pickle import dump,load
import pandas as pd
import numpy as np

app = Flask( __name__)

@app.route("/")
def root():
    return '<h1>asds</h1>'

@app.route("/shoot")
def formtest():
    return render_template('shoot_info.html')

def shooting(x,y,gd,sp,ms,ps,vs,pk,fk,cv,bc,blc,jh,clm,shoot_type,position):

    import itertools

    distance = np.sqrt((x - 1)**2 + (y - 0.5)**2)

    transform_values = []

    load_trainmm = pickle.load(open('./train_scaler.pkl', 'rb'))

    transform_values.append((load_trainmm.transform([[gd,sp,ms,ps,vs,pk,fk,cv,bc,blc,jh,clm]])).tolist()[0])

    transform_values = list(itertools.chain(*transform_values))

    transform_values.insert(0,distance)

    transform_values.insert(1,x)

    transform_values.insert(2,y)

    transform_values.append(shoot_type)

    transform_values.append(position)

    return transform_values

def shoot_predict(transform_values):

    input_df = pd.DataFrame(columns=['distance', 'x', 'y', '골 결정력', '슛 파워', '중거리 슛', '위치 선정', '발리 슛',
       '페널티 킥', '프리킥', '커브', '볼 컨트롤', '밸런스', '점핑 헤더', '침착성','type','포지션', 'type_1',
       'type_2', 'type_3', 'type_4', 'type_5', 'type_6', 'type_7', 'type_8',
       'type_9', 'type_10', '포지션_CAM', '포지션_CB', '포지션_CDM', '포지션_CF', '포지션_CM',
       '포지션_GK', '포지션_LB', '포지션_LM', '포지션_LW', '포지션_LWB', '포지션_RB', '포지션_RF',
       '포지션_RM', '포지션_RW', '포지션_RWB', '포지션_ST'])
        
    input_df = input_df.append(pd.Series(transform_values, index=['distance', 'x', 'y', '골 결정력', '슛 파워', '중거리 슛', '위치 선정', '발리 슛',
        '페널티 킥', '프리킥', '커브', '볼 컨트롤', '밸런스', '점핑 헤더', '침착성','type','포지션']),ignore_index=True)

    input_df.iloc[0,16+int(transform_values[15])] = 1 # 슈팅 타입 dummy 셀프 생성
    input_df.loc[0,'포지션_{}'.format(transform_values[16])] = 1 # 포지션 dummy 셀프 생성
    input_df = input_df.drop(['type','포지션'],axis=1)

    input_df = input_df.fillna(0)

    xgb_model = pickle.load(open('xgb_model.model', 'rb'))

    result = xgb_model.predict(input_df)[0]

    if result == 0:

        p = 'No Goal'

    else:
        p = 'Goal'

    return p


@app.route("/formproc")
def formproc():

    shoot_x = float(request.args['x'])
    shoot_y = float(request.args['y'])
    shoot_type = int(request.args['my_type'])
    position = request.args['pos']
    goal_decision = int(request.args['gd'])
    shoot_power = int(request.args['sp'])
    md_shoot = int(request.args['ms'])
    pos_sel = int(request.args['ps'])
    v_shoot = int(request.args['vs'])
    pk = int(request.args['pk'])
    fk = int(request.args['fk'])
    curve = int(request.args['curve'])
    ball_ctrl = int(request.args['bc'])
    balance = int(request.args['balance'])
    hj = int(request.args['hj'])
    calm = int(request.args['calm'])
    

    tv = shooting(shoot_x,shoot_y,goal_decision,shoot_power,md_shoot,pos_sel,v_shoot,pk,fk,curve,ball_ctrl,balance,hj,calm,shoot_type,position)
    result = shoot_predict(tv)
    
    # result = model.predict([[shoot_x,shoot_y,shoot_type,sp_grade,
    #             position,goal_decision,shoot_power,
    #             md_shoot,pos_sel,v_shoot,pk,fk,curve,
    #             ball_ctrl,balance,hj,calm]]).argmax(axis=1)
    return render_template('result.html', shoot_x=shoot_x,shoot_y=shoot_y, shoot_type=shoot_type, 
     position=position, goal_decision=goal_decision, shoot_power=shoot_power, md_shoot=md_shoot,
     pos_sel=pos_sel, v_shoot=v_shoot,pk=pk,fk=fk,curve=curve,ball_ctrl=ball_ctrl,balance=balance,hj=hj,calm=calm,result=result)




if __name__ == "__main__":
    # 웹 서버 구동
    app.run(host='localhost',port=4000, debug=True) # host: 모든 클라이언트 ip 주소 처리, port: 프로세스 식별자
