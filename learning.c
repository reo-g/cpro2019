#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "nn.h"
#include "MT.h" //疑似乱数生成としてMersenne Twisterを用いた
//"MT.h"は　http://www.sat.t.u-tokyo.ac.jp/~omi/code/MT.h より取得

#define M_PI 3.14159265358979323846 //Macでは定義が不要

//ReLU層の計算
void relu(int n,const float *x,float *y){
    for(int i=0;i<n;i++){
        y[i] = (x[i]>0) ? x[i] : 0;
    }
}
//fc層の計算
void fc(int m,int n,const float *x,const float *A,const float *b,float *y){
    for(int i=0;i<m;i++){
        y[i] = b[i];
        for(int j=0;j<n;j++){
            y[i] += A[n*i+j] * x[j];
        }
    }
}
//softmax層の計算
void softmax(int n,const float *x,float *y){
    float x_max = 0;
    float exp_sum = 0;
    for(int i=0;i<n;i++){
        if(x_max < x[i]){
            x_max = x[i];
        }   
    }
    for(int i=0;i<n;i++){
        exp_sum += exp(x[i]-x_max);
    }
    for(int i=0;i<n;i++){
        y[i] = exp(x[i]-x_max) / exp_sum;
    }
}
//6層NNの推論
int inference6(const float *A1,const float *A2,const float *A3,const float *b1,const float *b2,const float *b3,float *x,float *y){
    float *y1 = malloc(sizeof(float)*50);
    float *y2 = malloc(sizeof(float)*100);
    float temp = 0;
    int index;

    fc(50,784,x,A1,b1,y1);
    relu(50,y1,y1);
    fc(100,50,y1,A2,b2,y2);
    relu(100,y2,y2);
    fc(10,100,y2,A3,b3,y);
    softmax(10,y,y);

    for(int i=0;i<=9;i++){
        if(temp < y[i]){
            temp = y[i];
            index = i;
        }
    }
    free(y1);free(y2);
    return index;
}
//softmax層の誤差逆伝搬
void softmaxwithloss_bwd(int n,const float *y,unsigned char t,float *dEdx){
    for(int i=0;i<n;i++){
        dEdx[i] = (i==t)? y[i] - 1.0 : y[i];
    }
}
//ReLU層の誤差逆伝搬
void relu_bwd(int n,const float *x,const float *dEdy,float *dEdx){
    for(int i=0;i<n;i++){
        dEdx[i] = (x[i]>0)? dEdy[i] : 0;
    }
}
//fc層の誤差逆伝搬
void fc_bwd(int m,int n,const float *x,const float *dEdy,const float *A,float *dEdA,float *dEdb,float *dEdx){
    for(int i=0;i<m;i++){
        dEdb[i] = dEdy[i];
        for(int j=0;j<n;j++){
            dEdA[n*i+j] = dEdy[i] * x[j];
        }
    }

    for(int i=0;i<n;i++){
        dEdx[i] = 0;
        for(int j=0;j<m;j++){
            dEdx[i] += A[n*j+i] * dEdy[j];
        }
    }
}
//6層NNの誤差逆伝搬
void backward6(const float *A1,const float *A2,const float *A3,const float *b1,const float *b2,const float *b3,const float *x,unsigned char t,float *y,float *dEdA1,float *dEdA2,float *dEdA3,float *dEdb1,float *dEdb2,float *dEdb3){
    float *relu1_before = malloc(sizeof(float)*50);
    float *relu2_before = malloc(sizeof(float)*100);
    float *fc2_before = malloc(sizeof(float)*50);
    float *fc3_before = malloc(sizeof(float)*100);
    //順伝搬
    fc(50,784,x,A1,b1,relu1_before);
    relu(50,relu1_before,fc2_before);
    fc(100,50,fc2_before,A2,b2,relu2_before);
    relu(100,relu2_before,fc3_before);
    fc(10,100,fc3_before,A3,b3,y);
    softmax(10,y,y);

    float *dx3 = malloc(sizeof(float)*10);
    float *dx2 = malloc(sizeof(float)*100);
    float *dx1 = malloc(sizeof(float)*50);
    float *dx0 = malloc(sizeof(float)*784);
    //逆伝搬
    softmaxwithloss_bwd(10,y,t,dx3);
    fc_bwd(10,100,fc3_before,dx3,A3,dEdA3,dEdb3,dx2);
    relu_bwd(100,relu2_before,dx2,dx2);
    fc_bwd(100,50,fc2_before,dx2,A2,dEdA2,dEdb2,dx1);
    relu_bwd(50,relu1_before,dx1,dx1);
    fc_bwd(50,784,x,dx1,A1,dEdA1,dEdb1,dx0);

    free(relu1_before);free(relu2_before);
    free(fc2_before);free(fc3_before);
    free(dx0);free(dx1);free(dx2);free(dx3);
}
//indexをランダムシャッフル
void shuffle(int n,int *x){
    int t = 0;
    for(int i=0;i<n;i++){
        t = genrand_int32() % n;
        int temp = x[i];
        x[i] = x[t];
        x[t] = temp;
    }
}
//クロスエントロピー誤差を計算
float cross_entropy_error(const float *y,int t){
    return -1*log(y[t]+1e-7);
}
//行列の和を計算
void add(int n,const float *x,float *o){
    for (int i = 0; i < n;i++){
        o[i] = x[i] + o[i];
    }
}
//行列のスカラー積を計算
void scale(int n,float x,float *o){
    for (int i = 0; i < n;i++){
        o[i] = o[i] * x;
    }
}
//同一データでの行列の初期化
void init(int n,float x,float *o){
    for (int i = 0; i < n;i++){
        o[i] = x;
    }
}
//rand関数を用いて[-1:1]の一様分布で初期化
void rand_init(int n,float *o){
    for (int i = 0; i < n;i++){
        o[i] = (float)(rand() - (RAND_MAX / 2)) / (RAND_MAX / 2); //[-1:1]
    }
}

//ボックスミュラー法を用いた正規分布に従う疑似乱数の生成
double rand_normal( double mu, double sigma ){
    double z=sqrt( -2.0*log(genrand_real3())) * sin( 2.0*M_PI*genrand_real3());
    return mu + sigma*z;
}
//Heの初期値
void he_init(int n,float *o){
    for(int i=0;i<n;i++){
        o[i]=rand_normal(0,sqrt(2.0/n));
    }
}

//Xavierの初期値
void xavier_init(int n,float *o){
    for(int i=0;i<n;i++){
        o[i]=rand_normal(0,sqrt(1.0/n));
    }
}

//標準偏差0.01のガウス分布に従う重みの初期化
void normal_init(int n,float *o){
    for(int i=0;i<n;i++){
        o[i]=rand_normal(0,0.01);
    }
}

//Optimizer:Momentum SGD
void momentum(int n,float *v,float *o,float *t){
    float mu = 0.9;
    float r = 0.01;
    for(int i=0;i<n;i++){
        v[i] = mu * v[i] - r * o[i];
    }
    add(n,v,t);
}
//Optimizer:Adam
void Adam(int n,float *m,float *v,float *i,float *o,float t){
    float beta1 = 0.9;
    float beta2 = 0.999;
    float alpha = 0.001;
    float *mh = malloc(sizeof(float)*n);
    float *vh = malloc(sizeof(float)*n);
    for(int j=0;j<n;j++){
        m[j] = beta1*m[j]+(1-beta1)*i[j];
        v[j] = beta2*v[j]+(1-beta2)*i[j]*i[j];

        mh[j] = m[j] / (1-pow(beta1,t));
        vh[j] = v[j] / (1-pow(beta2,t));

        o[j] = o[j] -alpha*mh[j]/(sqrt(vh[j])+1e-7);
    }
    free(mh);free(vh);
}
//Optimizer:AdaGrad
void AdaGrad(int n,float *h,float *i,float *o){
    float alpha = 0.001;
    float *ht = malloc(sizeof(float)*n);
    for(int j=0;j<n;j++){
        h[j] = h[j]+ i[j] * i[j];
        ht[j] = alpha / sqrt(h[j]);
        o[j] = o[j] - ht[j] * i[j];
    }
    free(ht);
}
//係数の保存
void save(const char *filename,int m,int n,const float *A,const float *b){
    FILE *fp;
    fp = fopen(filename,"wb");
    fwrite(A,sizeof(float),m*n,fp);
    fwrite(b,sizeof(float),m,fp);
    fclose(fp);
}

//行列データのコピー
void copy(int n,const float *x,float *o){
    for (int i = 0; i < n;i++){
        o[i] = x[i];
    }
}

int main(void)
{
    float *train_x = NULL;
    unsigned char *train_y = NULL;
    int train_count = -1;

    float *test_x = NULL;
    unsigned char *test_y = NULL;
    int test_count = -1;

    int width = -1;
    int height = -1;

    load_mnist(&train_x, &train_y, &train_count,
                &test_x, &test_y, &test_count,
                &width, &height);

    // これ以降，３層NN の係数 A_784x10 および b_784x10 と，
    // 訓練データ train_x + 784*i (i=0,...,train_count-1), train_y[0]～train_y[train_count-1],
    // テストデータ test_x + 784*i (i=0,...,test_count-1), test_y[0]～test_y[test_count-1],
    // を使用することができる．

    srand(time(NULL));
    int c1,c2; //入力確認用
    int epoch = 20;
    int batch_size = 100;
    int opt_select = 0;
    int init_select = 0;

    float Loss_ave = 0;
    float Accuracy = 0;
    //float Loss_ave1 = 0;
    //float Accuracy1 = 0;
    float max_accuracy = 0;

    float *y = malloc(sizeof(float)*10);

    int *index = malloc(sizeof(int)*train_count);

    float *A1 = malloc(sizeof(float)*784*50);
    float *b1 = malloc(sizeof(float)*50);
    float *A2 = malloc(sizeof(float)*50*100);
    float *b2 = malloc(sizeof(float)*100);
    float *A3 = malloc(sizeof(float)*100*10);
    float *b3 = malloc(sizeof(float)*10);

    float *best_A1 = malloc(sizeof(float)*784*50);
    float *best_b1 = malloc(sizeof(float)*50);
    float *best_A2 = malloc(sizeof(float)*50*100);
    float *best_b2 = malloc(sizeof(float)*100);
    float *best_A3 = malloc(sizeof(float)*100*10);
    float *best_b3 = malloc(sizeof(float)*10);

    float *dEdA1 = malloc(sizeof(float)*784*50);
    float *dEdb1 = malloc(sizeof(float)*50);
    float *dEdA2 = malloc(sizeof(float)*50*100);
    float *dEdb2 = malloc(sizeof(float)*100);
    float *dEdA3 = malloc(sizeof(float)*100*10);
    float *dEdb3 = malloc(sizeof(float)*10);

    float *dEdA1_ave = malloc(sizeof(float)*784*50);
    float *dEdb1_ave = malloc(sizeof(float)*50);
    float *dEdA2_ave = malloc(sizeof(float)*50*100);
    float *dEdb2_ave = malloc(sizeof(float)*100);
    float *dEdA3_ave = malloc(sizeof(float)*100*10);
    float *dEdb3_ave = malloc(sizeof(float)*10);
    //SGD 
    float learning_rate = 0.1;
    //Momentum SGD
    float *vA1 = malloc(sizeof(float)*784*50);
    float *vb1 = malloc(sizeof(float)*50);
    float *vA2 = malloc(sizeof(float)*50*100);
    float *vb2 = malloc(sizeof(float)*100);
    float *vA3 = malloc(sizeof(float)*100*10);
    float *vb3 = malloc(sizeof(float)*10);
    //AdaGrad
    float *hA1 = malloc(sizeof(float)*784*50);
    float *hb1 = malloc(sizeof(float)*50);
    float *hA2 = malloc(sizeof(float)*50*100);
    float *hb2 = malloc(sizeof(float)*100);
    float *hA3 = malloc(sizeof(float)*100*10);
    float *hb3 = malloc(sizeof(float)*10);    
    //Adam
    float t = 1.0;

    float *b_vA1 = malloc(sizeof(float)*784*50);
    float *b_vb1 = malloc(sizeof(float)*50);
    float *b_vA2 = malloc(sizeof(float)*50*100);
    float *b_vb2 = malloc(sizeof(float)*100);
    float *b_vA3 = malloc(sizeof(float)*100*10);
    float *b_vb3 = malloc(sizeof(float)*10);

    float *b_mA1 = malloc(sizeof(float)*784*50);
    float *b_mb1 = malloc(sizeof(float)*50);
    float *b_mA2 = malloc(sizeof(float)*50*100);
    float *b_mb2 = malloc(sizeof(float)*100);
    float *b_mA3 = malloc(sizeof(float)*100*10);
    float *b_mb3 = malloc(sizeof(float)*10);
    //初期化に使用する乱数分布の選択
    do{
        printf("初期値を選択してください\n一様分布:1 ガウス分布(S.D.=0.01):2 Heの初期値:3 Xavierの初期値:4\nYour select:");
        c1 = scanf("%d",&init_select);
        if(c1 != 1){
            printf("Input Error!\n");
            scanf("%*s");
        }
    }while(c1!=1||init_select<1||init_select>4);
    //学習に使用するOptimizerの選択
    do{
        printf("\nOptimizerを選択してください\nSGD:1 MomentumSGD:2 Adagrad:3 Adam:4\nYour select:");
        c2 = scanf("%d",&opt_select);
        if(c2 != 1){
            printf("Input Error!\n");
            scanf("%*s");
        }
    }while(c2!=1||opt_select<1||opt_select>4);
    printf("\n");
    
    //Optimizerの選択
    switch (opt_select)
    {
    case 1:{ //SGD
        free(vA1);free(vA2);free(vA3);free(vb1);free(vb2);free(vb3);
        free(hA1);free(hA2);free(hA3);free(hb1);free(hb2);free(hb3);
        free(b_vA1);free(b_vA2);free(b_vA3);free(b_vb1);free(b_vb2);free(b_vb3);
        free(b_mA1);free(b_mA2);free(b_mA3);free(b_mb1);free(b_mb2);free(b_mb3);
        break;
        }
    case 2:{ //Momentum SGD
        init(784*50,0,vA1);
        init(50,0,vb1);
        init(50*100,0,vA2);
        init(100,0,vb2);
        init(100*10,0,vA3);
        init(10,0,vb3);
        free(hA1);free(hA2);free(hA3);free(hb1);free(hb2);free(hb3);
        free(b_vA1);free(b_vA2);free(b_vA3);free(b_vb1);free(b_vb2);free(b_vb3);
        free(b_mA1);free(b_mA2);free(b_mA3);free(b_mb1);free(b_mb2);free(b_mb3);
        break;
        }
    case 3:{ //Adagrad
        init(784*50,1e-7,hA1);
        init(50,1e-7,hb1);
        init(50*100,1e-7,hA2);
        init(100,1e-7,hb2);
        init(100*10,1e-7,hA3);
        init(10,1e-7,hb3);
        free(vA1);free(vA2);free(vA3);free(vb1);free(vb2);free(vb3);
        free(b_vA1);free(b_vA2);free(b_vA3);free(b_vb1);free(b_vb2);free(b_vb3);
        free(b_mA1);free(b_mA2);free(b_mA3);free(b_mb1);free(b_mb2);free(b_mb3);
        break;
        }
    case 4:{ //Adam
        init(784*50,0,b_vA1);
        init(50,0,b_vb1);
        init(50*100,0,b_vA2);
        init(100,0,b_vb2);
        init(100*10,0,b_vA3);
        init(10,0,b_vb3);

        init(784*50,0,b_mA1);
        init(50,0,b_mb1);
        init(50*100,0,b_mA2);
        init(100,0,b_mb2);
        init(100*10,0,b_mA3);
        init(10,0,b_mb3);
        free(vA1);free(vA2);free(vA3);free(vb1);free(vb2);free(vb3);
        free(hA1);free(hA2);free(hA3);free(hb1);free(hb2);free(hb3);
        break;  
        }      
    }

    switch (init_select) //初期値の選択
    {
    case 1:{ //rand関数による一様分布
        rand_init(784*50,A1);
        rand_init(50,b1);
        rand_init(50*100,A2);
        rand_init(100,b2);
        rand_init(100*10,A3);
        rand_init(10,b3);
        break;
        }
    case 2:{ //標準偏差0.01のガウス分布
        normal_init(784*50,A1);
        normal_init(50,b1);
        normal_init(50*100,A2);
        normal_init(100,b2);
        normal_init(100*10,A3);
        normal_init(10,b3);
        break;
        }
    case 3:{ //Heの初期値
        he_init(784*50,A1);
        he_init(50,b1);
        he_init(50*100,A2);
        he_init(100,b2);
        he_init(100*10,A3);
        he_init(10,b3);
        break;
        }
    case 4:{ //Xavierの初期値
        xavier_init(784*50,A1);
        xavier_init(50,b1);
        xavier_init(50*100,A2);
        xavier_init(100,b2);
        xavier_init(100*10,A3);
        xavier_init(10,b3);
        break;
        }
    }

    for(int i=0;i<epoch;i++){
        for(int o=0;o<train_count;o++){
            index[o] = o;
        }
        shuffle(train_count,index);


        for(int j=0;j<train_count/batch_size;j++){
            init(784*50,0,dEdA1_ave);
            init(50,0,dEdb1_ave);
            init(50*100,0,dEdA2_ave);
            init(100,0,dEdb2_ave);
            init(100*10,0,dEdA3_ave);
            init(10,0,dEdb3_ave);
            
            for(int k=0;k<batch_size;k++){
                printf("\rEpoch%3d:[%3d/100%%]",i,((k + 1 + batch_size * j) * 100/ train_count));
                backward6(A1,A2,A3,b1,b2,b3,train_x + 784*index[batch_size*j+k],
                    train_y[index[batch_size*j+k]],y,dEdA1,dEdA2,dEdA3,dEdb1,dEdb2,dEdb3);
                add(784*50,dEdA1,dEdA1_ave);
                add(50,dEdb1,dEdb1_ave);
                add(50*100,dEdA2,dEdA2_ave);
                add(100,dEdb2,dEdb2_ave);
                add(100*10,dEdA3,dEdA3_ave);
                add(10,dEdb3,dEdb3_ave);
            }
            scale(784*50,1/((float)batch_size),dEdA1_ave);
            scale(50,1/((float)batch_size),dEdb1_ave);
            scale(50*100,1/((float)batch_size),dEdA2_ave);
            scale(100,1/((float)batch_size),dEdb2_ave);
            scale(100*10,1/((float)batch_size),dEdA3_ave);
            scale(10,1/((float)batch_size),dEdb3_ave);

            switch (opt_select)
            {
            case 1:{ //SGD
                scale(784*50,-learning_rate,dEdA1_ave);
                scale(50,-learning_rate,dEdb1_ave);
                scale(50*100,-learning_rate,dEdA2_ave);
                scale(100,-learning_rate,dEdb2_ave);
                scale(100*10,-learning_rate,dEdA3_ave);
                scale(10,-learning_rate,dEdb3_ave);
                add(784*50,dEdA1_ave,A1);
                add(50,dEdb1_ave,b1);
                add(50*100,dEdA2_ave,A2);
                add(100,dEdb2_ave,b2);
                add(100*10,dEdA3_ave,A3);
                add(10,dEdb3_ave,b3);
                break;
                }
            case 2:{ //Momentum SGD
                momentum(784*50,vA1,dEdA1_ave,A1);
                momentum(50,vb1,dEdb1_ave,b1);
                momentum(50*100,vA2,dEdA2_ave,A2);
                momentum(100,vb2,dEdb2_ave,b2);
                momentum(100*10,vA3,dEdA3_ave,A3);
                momentum(10,vb3,dEdb3_ave,b3);
                break;
                }
            case 3:{ //Adagrad
                AdaGrad(784*50,hA1,dEdA1_ave,A1);
                AdaGrad(50,hb1,dEdb1_ave,b1);
                AdaGrad(50*100,hA2,dEdA2_ave,A2);
                AdaGrad(100,hb2,dEdb2_ave,b2);
                AdaGrad(100*10,hA3,dEdA3_ave,A3);
                AdaGrad(10,hb3,dEdb3_ave,b3);
                break;
                }
            case 4:{ //Adam
                Adam(784*50,b_mA1,b_vA1,dEdA1_ave,A1,t);
                Adam(50,b_mb1,b_vb1,dEdb1_ave,b1,t);
                Adam(50*100,b_mA2,b_vA2,dEdA2_ave,A2,t);
                Adam(100,b_mb2,b_vb2,dEdb2_ave,b2,t);
                Adam(100*10,b_mA3,b_vA3,dEdA3_ave,A3,t);
                Adam(10,b_mb3,b_vb3,dEdb3_ave,b3,t);
                t++;   
                break;
                }
            }
        
        } 

        float sum = 0;
        float loss_sum = 0;
        //float sum_train = 0;
        //float loss_train = 0;
        for(int m=0;m<test_count;m++){
            if(inference6(A1,A2,A3,b1,b2,b3,test_x+m*784,y) == test_y[m]){
                sum++;
            }
            loss_sum += cross_entropy_error(y,test_y[m]);
        }
        /* for(int m=0;m<train_count;m++){
            if(inference6(A1,A2,A3,b1,b2,b3,train_x+m*784,y) == train_y[m]){
                sum_train++;
            }
            loss_train += cross_entropy_error(y,train_y[m]);
        }
        */
        printf("\nLoss Average: %f (%+.3f)\n",loss_sum/test_count,loss_sum/test_count-Loss_ave);
        printf("Accuracy: %f (%+5.2f)\n",sum*100.0/test_count,sum*100.0/test_count-Accuracy);
        //printf("Loss Average(train): %f (%+.3f)\n",loss_train/train_count,loss_train/train_count-Loss_ave1);
        //printf("Accuracy(train): %f (%+.2f)\n",sum_train*100.0/train_count,sum_train*100.0/train_count-Accuracy1);
        
        //ここまでで最良のモデルの行列を記録しておく
        if(sum*100.0/test_count > max_accuracy){
            max_accuracy = sum*100.0/test_count;
            copy(784*50,A1,best_A1);
            copy(50,b1,best_b1);
            copy(50*100,A2,best_A2);
            copy(100,b2,best_b2);
            copy(100*10,A3,best_A3);
            copy(10,b3,best_b3);
        }
        printf("Max Accuracy: %.2f%%\n\n",max_accuracy+0.001);
        Loss_ave = loss_sum/test_count;
        Accuracy = sum*100.0/test_count;
        //Loss_ave1 = loss_train/train_count;
        //Accuracy1 = sum_train*100.0/train_count;  
    }
    //正解率が最も高かったモデルの行列を保存する
    save("fc1.dat",50,784,best_A1,best_b1);
    save("fc2.dat",100,50,best_A2,best_b2);
    save("fc3.dat",10,100,best_A3,best_b3);
    
    printf("finish!\n");

  return 0;
}
