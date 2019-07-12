#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include "nn.h"

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
//データの読み込み
void load(const char *filename,int m,int n,float *A,float *b){
    FILE *fp;
    fp = fopen(filename, "rb");
    fread(A,sizeof(float),m*n, fp);
    fread(b,sizeof(float),m, fp);
    fclose(fp);
}

int main(int argc, char const *argv[])
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

    //係数の初期化
    float *A1 = malloc(sizeof(float)*784*50);
    float *A2 = malloc(sizeof(float)*50*100);
    float *A3 = malloc(sizeof(float)*100*10);
    float *b1 = malloc(sizeof(float)*50);
    float *b2 = malloc(sizeof(float)*100);
    float *b3 = malloc(sizeof(float)*10);
    float *y = malloc(sizeof(float)*10);

    
    //文法確認
    if (argc < 5){
        printf("inference fc1.dat fc2.dat fc3.dat pic.bmpの形式で実行してください\n");
        return -1;
    }
    //行列データを読み込んで代入
    load(argv[1], 50, 784, A1, b1);
    load(argv[2], 100, 50, A2, b2);
    load(argv[3], 10, 100, A3, b3);

    //画像ファイルの読み込み
    float *x = load_mnist_bmp(argv[4]);

    int predict = 0;
    predict = inference6(A1, A2, A3, b1, b2, b3, x ,y);

    //それぞれの数字に対する予測した確率を表示させる
    printf("  0    1    2    3    4    5    6    7    8    9  \n");
    for(int i=0;i<10;i++){
        printf("%3d%% ",(int)(y[i]*100.0));
    }
    printf("\nPicture's Number is %d.\n", predict);

  return 0;
}

