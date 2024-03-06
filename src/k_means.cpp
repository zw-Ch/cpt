//
// Created by ASUS on 2024/1/30.
//
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <fstream>
#include "k_means.h"

using namespace std;

#define K 3  // 簇数量
#define N 50 // 点数量
#define D 2  // 维度

double point[N][D];
double barycenter_initial[K][D];  // 初始，质心位置
double barycenter_before[K][D];   // 变换前，质心位置
double barycenter_finished[K][D]; // 最终，质心位置
double O_Distance[K];
int belongWhichBC[N];
double mid[D];

// 初始化数据点
void CoordinateDistribution(int n, int d)
{
    srand((unsigned)time(NULL)); // 保证随机性
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < d; j++)
        {
            point[i][j] = rand() % 101;
        }
    }
}

// 初始化质心
void initBarycenter(int k, int d)
{
    for (int i = 0; i < k; i++)
    {
        for (int j = 0; j < d; j++)
        {
            point[i][j] = rand() % 101;
        }
    }
}

void k_means()
{
    // 为N个点随机分配D维度的坐标
    int n = N, d = D, k = K;
    CoordinateDistribution(n, d);

    cout << "簇数量 K = " << K << endl
         << "点数量 N = " << N << endl
         << "维度 D = " << D << endl
         << endl;

    cout << "系统生成的N个点如下：" << endl;
    for (int i = 0; i < n; i++)
    {
        cout << "第" << i + 1 << "个"
             << "\t";
        for (int j = 0; j < d; j++)
        {
            cout << point[i][j] << "\t";
        }
        cout << endl;
    }
    cout << endl;

    cout << "系统生成的初始簇心如下：" << endl;
    for (int i = 0; i < n; i++)
    {
        cout << "第" << i + 1 << "个"
             << "\t";
        for (int j = 0; j < d; j++)
        {
            cout << barycenter_initial[i][j] << "\t";
        }
        cout << endl;
    }
    cout << endl;

    for (int i = 0; i < k; i++) {
        for (int j = 0; j < d; j ++) {
            barycenter_before[i][j] = barycenter_initial[i][j];
            barycenter_finished[i][j] = -1;
        }
    }

    int times = 0;
    while (true) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < k; j++) {
                double sum = 0;
                for (int x = 0; x < d; x++) {
                    sum = sum + pow(point[i][x] - barycenter_before[j][x], 2);
                }
                O_Distance[j] = sum;
            }
            int x = 0, temp = x;
            while (x < k) {
                if (O_Distance[x] < O_Distance[temp]) {
                    temp = x;
                    x ++;
                } else {
                    x ++;
                }
            }
            belongWhichBC[i] = temp;
        }

        for (int i = 0; i < d; i++) {
            mid[i] = 0;
        }
    }
}
