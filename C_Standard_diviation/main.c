//
//  main.c
//  C_Standard_diviation
//
//  Created by PARK JAICHANG on 7/22/16.
//  Copyright © 2016 JAICHANGPARK. All rights reserved.
//
//  표준편차에대한 sisd, simd 비교 연산 처리 프로그램
//  mac achi 이기 때문에 window와 차이가 발생할 수 있다.
//  100만개의 사이즈

/*-----------------------------------------------------
 표준 편차 : roots((1/n)*sum((x-x_avg)^2))
-----------------------------------------------------*/

#include <stdio.h>
#include <sys/time.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>
#include <xmmintrin.h>
#include <emmintrin.h>

#define ARRAYSIZE 10000

struct timeval start, stop ;
double result_time = 0;
int i = 0;

double timedifference_msec();
float GetStDev(float* Array, int Size);
float GetStDevIntrinsic(float *Array_intric, int Size_intric);

int main(int argc, const char * argv[]) {

    float Array[ARRAYSIZE] = {0};
    for (int i = 0; i<ARRAYSIZE; i++) {
        Array[i] = i;
    }
    double result ;
    
    gettimeofday(&start, NULL);
    result = GetStDev(Array, ARRAYSIZE);
    gettimeofday(&stop, NULL);
    result_time = timedifference_msec(stop, start);
    printf("C:STDEV RESULT : %f, code executed in %f microsecond \n", result,result_time);
    
    gettimeofday(&start, NULL);
    result = GetStDevIntrinsic(Array, ARRAYSIZE);
    gettimeofday(&stop, NULL);
    result_time = timedifference_msec(stop, start);
    printf("C_intric:STDEV RESULT : %f, code executed in %f microsecond \n", result,result_time);
    

    return 0;
}

float GetStDev(float* Array, int Size){
    
    float sum = 0 ; //총합을 구하게 된다.
    for(i = 0; i<Size; i++){
        
        sum += Array[i];
    }
    
    float Average = sum / Size ; // 평균값을 계산하는 식
    float DistValue = 0 ;
    float SumDist = 0 ;
    
    for(i = 0 ; i < Size; i++){

        DistValue = Average - Array[i];
        SumDist += DistValue*DistValue;
        
    }
    return sqrt(SumDist/Size);
}
float GetStDevIntrinsic(float *Array_intric, int Size_intric){
    
    int LoopSize = (Size_intric / 4) * 4 ;
    float Sum_intric = 0;
    
    //  float Result_intric = 0;
    __m128 xmmSum = _mm_setzero_ps();
    __m128 xmmCur = _mm_setzero_ps();
    __m128 xmmResult = _mm_setzero_ps();
    
    for (i = 0 ; i < LoopSize; i += 4) {
        xmmCur = _mm_loadu_ps(Array_intric+i);  //4개씩 연속으로 xmmCur에 읽어들인다.
        xmmSum = _mm_add_ps(xmmSum, xmmCur);    // 4개씩 합을 구한다.
    }
    //  4개의 병령 합을 하나로 구한다.
    xmmResult = _mm_shuffle_ps(xmmResult, xmmSum, 0x40); // ! _mm_shuffle_ps !
    xmmSum = _mm_add_ps(xmmSum, xmmResult);
    xmmResult = _mm_shuffle_ps(xmmResult, xmmSum, 0x30);
    xmmSum = _mm_add_ps(xmmSum, xmmResult);
    xmmSum = _mm_shuffle_ps(xmmSum, xmmSum, 2);
    _mm_store_ss(&Sum_intric, xmmSum);
    
    //  4배수 값 이외의 나머지 값을 더해준다.
    for (i = LoopSize; i<Size_intric; i++) {
        Sum_intric += Array_intric[i];
    }
    
    //  평균 값을 구한다.
    float Average_intric = Sum_intric / Size_intric;
    
    //  편차 계산을 위한 4개의 평균 데이터를 생성
    //  xmmAVG : 위에서 계산된 평균 값이 4개의 pack으로 저장된다.
    //  xmmDistribution(분포), xmmSumDist : 0으로 초기화
    __m128 xmmAVG = _mm_set1_ps(Average_intric);
    __m128 xmmDistribution = _mm_setzero_ps();
    __m128 xmmSumDist = _mm_setzero_ps();
    
    for (i = 0; i<LoopSize; i +=4) {
        xmmCur = _mm_loadu_ps(Array_intric+i); // 4개씩 읽어온다. 128/32 = 4
        xmmDistribution = _mm_sub_ps(xmmAVG, xmmCur);   // 평균값에서 뺄셈 (x-x_avg)
        xmmDistribution = _mm_mul_ps(xmmDistribution, xmmDistribution); // 제곱
        xmmSumDist = _mm_add_ps(xmmSumDist, xmmDistribution);   // 편차의 합
    }
    
    xmmResult = _mm_setzero_ps();
    Sum_intric = 0;
    
    //  4개의 병렬 데이터의 합을 하나로..
    xmmResult =_mm_shuffle_ps(xmmResult, xmmSumDist, 0x40);
    xmmSumDist = _mm_add_ps(xmmSumDist, xmmResult);
    xmmResult = _mm_shuffle_ps(xmmResult, xmmSumDist, 0x30);
    xmmSumDist = _mm_add_ps(xmmSumDist, xmmResult);
    xmmSumDist = _mm_shuffle_ps(xmmSumDist, xmmSumDist, 2);
    _mm_store_ss(&Sum_intric, xmmSumDist);
    
    //  나머지를 구하기 위하여 다음 반복문이 사용된다.
    for (i=LoopSize; i<Size_intric; i++) {
        Sum_intric += (Average_intric-Array_intric[i])*(Average_intric-Array_intric[i]);
    }
    //  제곱근 값을 return..!
    return sqrt(Sum_intric/Size_intric);
}
double timedifference_msec(struct timeval t0, struct timeval t1){
    
    return (double)(t0.tv_usec - t1.tv_usec) / 1000000 + (double)(t0.tv_sec - t1.tv_sec);
}
