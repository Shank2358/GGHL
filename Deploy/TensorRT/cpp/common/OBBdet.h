#include <iostream>
#include <fstream>
#include <vector>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <math.h>
#include<algorithm>
#include <opencv2/opencv.hpp>
#define maxn 51
const double eps = 1E-8;
using namespace std;
int sig(double d)
{
    return (d > eps) - (d < -eps);
}
typedef struct OBBInfo
{
    OBBInfo(float x1, float y1, float x2, float y2, float conf, float pred_r, vector<float> pred_s, vector<float> prob) : x1(x1), y1(y1), y2(y2), conf(conf), pred_r(pred_r), pred_s(pred_s), prob(prob) {}
    //委托构造函数
    OBBInfo() : OBBInfo(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, vector<float>(4, 3.0), vector<float>(15, 0.0)) {}
    //需要写一个构造函数
    float x1;
    float y1;
    float x2;
    float y2;
    float conf;
    float pred_r;
    vector<float> pred_s;
    vector<float> prob;
} OBBInfo;
typedef struct Point
{
    double x, y;
    Point() {}
    Point(float x, float y) : x(x), y(y) {}
    bool operator==(const Point &p) const
    {
        return sig(x - p.x) == 0 && sig(y - p.y) == 0;
    }
}Point;
typedef struct OBBInfo8
{
    OBBInfo8(float x1, float y1, float x2, float y2, float x3, float y3, float x4, float y4, float conf, vector<float> prob,int label) : x1(x1), y1(y1), x2(x2),
    y2(y2),x3(x3),y3(y3),x4(x4),y4(y4),conf(conf),prob(prob),label(label){}
    //委托构造函数的处理
    OBBInfo8():OBBInfo8(0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,vector<float>(15,0.0),0){}
    float x1;
    float y1;
    float x2;
    float y2;
    float x3;
    float y3;
    float x4;
    float y4;
    float conf;
    vector<float> prob;
    int label;
} OBBInfo8;

typedef struct OBBInfo4
{
    float x1;
    float y1;
    float x2;
    float y2;
    float area;
    float conf;
    float get(string s)
    {
        if (s == "x1")
            return x1;
        else if (s == "x2")
            return x2;
        else if (s == "y1")
            return y1;
        else 
        {
            return y2;
        }
    }
} OBBInfo4;
float max_prob(vector<float> prob)
{
    float value = 0.0;
    for(auto i:prob)
    {
        value = max(i, value);
    }
    return value;
}
double cross(Point o, Point a, Point b)
{ //叉积
    return (a.x - o.x) * (b.y - o.y) - (b.x - o.x) * (a.y - o.y);
}
double area(Point *ps, int n)
{
    ps[n] = ps[0];
    double res = 0;
    for (int i = 0; i < n; i++)
    {
        res += ps[i].x * ps[i + 1].y - ps[i].y * ps[i + 1].x;
    }
    return res / 2.0;
}
int lineCross(Point a, Point b, Point c, Point d, Point &p)
{
    double s1, s2;
    s1 = cross(a, b, c);
    s2 = cross(a, b, d);
    if (sig(s1) == 0 && sig(s2) == 0)
        return 2;
    if (sig(s2 - s1) == 0)
        return 0;
    p.x = (c.x * s2 - d.x * s1) / (s2 - s1);
    p.y = (c.y * s2 - d.y * s1) / (s2 - s1);
    return 1;
}
void polygon_cut(Point *p, int &n, Point a, Point b, Point *pp)
{
    //    static Point pp[maxn];
    int m = 0;
    p[n] = p[0];
    for (int i = 0; i < n; i++)
    {
        if (sig(cross(a, b, p[i])) > 0)
            pp[m++] = p[i];
        if (sig(cross(a, b, p[i])) != sig(cross(a, b, p[i + 1])))
            lineCross(a, b, p[i], p[i + 1], pp[m++]);
    }
    n = 0;
    for (int i = 0; i < m; i++)
        if (!i || !(pp[i] == pp[i - 1]))
            p[n++] = pp[i];
    while (n > 1 && p[n - 1] == p[0])
        n--;
}
double intersectArea(Point a, Point b, Point c, Point d)
{
    Point o(0, 0);
    int s1 = sig(cross(o, a, b));
    int s2 = sig(cross(o, c, d));
    if (s1 == 0 || s2 == 0)
        return 0.0; //退化，面积为0
    if (s1 == -1)
        swap(a, b);
    if (s2 == -1)
        swap(c, d);
    Point p[10] = {o, a, b};
    int n = 3;
    Point pp[maxn];
    polygon_cut(p, n, o, c, pp);
    polygon_cut(p, n, c, d, pp);
    polygon_cut(p, n, d, o, pp);
    double res = fabs(area(p, n));
    if (s1 * s2 == -1)
        res = -res;
    return res;
}
double intersectArea(Point *ps1, int n1, Point *ps2, int n2)
{
    if (area(ps1, n1) < 0)
        reverse(ps1, ps1 + n1);
    if (area(ps2, n2) < 0)
        reverse(ps2, ps2 + n2);
    ps1[n1] = ps1[0];
    ps2[n2] = ps2[0];
    double res = 0;
    for (int i = 0; i < n1; i++)
    {
        for (int j = 0; j < n2; j++)
        {
            res += intersectArea(ps1[i], ps1[i + 1], ps2[j], ps2[j + 1]);
        }
    }
    return res; // assumeresispositive!
}
double iou_poly(vector<double> p, vector<double> q)
{
    Point ps1[maxn], ps2[maxn];
    int n1 = 4;
    int n2 = 4;
    for (int i = 0; i < 4; i++)
    {
        ps1[i].x = p[i * 2];
        ps1[i].y = p[i * 2 + 1];

        ps2[i].x = q[i * 2];
        ps2[i].y = q[i * 2 + 1];
    }
    double inter_area = intersectArea(ps1, n1, ps2, n2);
    double union_area = fabs(area(ps1, n1)) + fabs(area(ps2, n2)) - inter_area;
    double iou = inter_area / union_area;

    //    cout << "inter_area:" << inter_area << endl;
    //    cout << "union_area:" << union_area << endl;
    //    cout << "iou:" << iou << endl;

    return iou;
}
vector<OBBInfo> convert_result(vector<float>original_res)
{
    //将原来的单维度转换成对应的
    vector<OBBInfo> convert_res;
    int num_proposal = original_res.size() / 29;
    for (int i = 0; i < original_res.size();i=i+29)
    {
        OBBInfo cur_Box;
        cur_Box.x1 = (original_res[i] - original_res[i + 2])/2 ;
        cur_Box.y1 = (original_res[i + 1] - original_res[i + 3])/2 ;
        cur_Box.x2 = (original_res[i + 0]+original_res[i+2])/2;
        cur_Box.y2 = (original_res[i + 1] + original_res[i + 3])/2;
        cur_Box.conf = original_res[i + 13];//获取当前的置信度的处理的方式
        cur_Box.pred_s.assign(original_res.begin() + i + 4, original_res.begin() + i + 8);
        cur_Box.pred_r = original_res[i + 8];
        cur_Box.prob.assign(original_res.begin() + i + 14, original_res.begin() + i + 29);
        convert_res.push_back(cur_Box);
    }
    return convert_res;
}
vector<OBBInfo8> convert_pred(vector<OBBInfo>convert_result,const int test_size,const int ori_shape,
const float conf_threshs)
{
    float conf_thresh = conf_threshs;
    int org_h = ori_shape;
    int org_w = ori_shape;
    float resize_ratio = min(1.0 * test_size / org_w, 1.0 * test_size / org_h);
    float dw = (test_size - resize_ratio * org_w) / 2.0;
    float dh = (test_size - resize_ratio * org_h) / 2.0;
    vector<OBBInfo8> filter_Boxes;
    for (auto &i : convert_result)
    {
        i.x1 = 1.0 * (i.x1 - dw) / resize_ratio;
        i.x2 = 1.0 * (i.x2 - dw) / resize_ratio;
        i.y1 = 1.0 * (i.y1 - dh) / resize_ratio;
        i.y2 = 1.0 * (i.y2 - dh) / resize_ratio;
        if(i.pred_r>0.9)
        {
            i.pred_s.assign(4, 0.0);//如果这个值大于0.9 直接分配成0，否则不予分配
        }
        //超出原图的部分,进行对应的处理
        if(i.x1<0)
            i.x1 = 0;
        if(i.y1<0)
            i.y1 = 0;
        if(i.x2>org_w-1)
            i.x2 = org_w - 1;
        if(i.y2>org_h-2)
            i.y2 = org_h - 1;
        //第二次筛选，大小不对的话，直接给坐标赋值为0，即可
        if((i.x2<i.x1)||(i.y2<i.y1))
        {
            i.x1 = 0.0;
            i.x2 = 0.0;
            i.y1 = 0.0;
            i.y2 = 0.0;
            i.pred_s.assign(4,0.0);      
        }
    }
    //将缩放的补充完成后进行下一步的处理
    //下一步筛选的过程，一个
    for (int i = 0; i < convert_result.size();i++)
    {
        //获取scale_mask的处理方式
        OBBInfo cur_box = convert_result[i];//获取当前BBox,然后进行对应的处理
        double area = sqrt((cur_box.y2 - cur_box.y1) * (cur_box.x2 - cur_box.x1));
        float scores = cur_box.conf * max_prob(cur_box.prob);
        if(area>0&&(scores>conf_thresh))
        {
            //这边需要进行一个转换
            OBBInfo8 bbox;
            bbox.x1 = cur_box.pred_s[0] * (cur_box.x2 - cur_box.x1) + cur_box.x1;
            bbox.y1 = cur_box.y1;
            bbox.x2 = cur_box.x2;
            bbox.y2 = cur_box.pred_s[1] * (cur_box.y2 - cur_box.y1) + cur_box.y1;
            bbox.x3 = cur_box.x2-cur_box.pred_s[2]*(cur_box.x2-cur_box.x1);
            bbox.y3 = cur_box.y2;
            bbox.x4 = cur_box.x1;
            bbox.y4 = cur_box.y2 - cur_box.pred_s[3] * (cur_box.y2 - cur_box.y1);
            bbox.conf = cur_box.conf;
            bbox.prob.assign(cur_box.prob.begin(), cur_box.prob.end());
            filter_Boxes.push_back(bbox);//如果符合这两个条件
        }
    }
    //转换成当前的shape 仅仅支持单张图推理 过滤bb后的图片
    return filter_Boxes;
}
bool Conf_compare(OBBInfo8 box1,OBBInfo8 box2)
{
    return box1.conf > box2.conf;
}
vector<OBBInfo4> Convert_8Points(vector<OBBInfo8> predict)
{
    vector<OBBInfo4> BBox_4;
    for (int i = 0; i < predict.size(); i++)
    {
        OBBInfo4 cur_box;
        float xmin = 10000;
        float ymin = 10000;
        float xmax = -1.0;
        float ymax = -1.0;
        xmin = min(predict[i].x1, xmin);
        xmin = min(predict[i].x2, xmin);
        xmin = min(predict[i].x3, xmin);
        xmin = min(predict[i].x4, xmin);
        xmax = max(predict[i].x1, xmax);
        xmax = max(predict[i].x2, xmax);
        xmax = max(predict[i].x3, xmax);
        xmax = max(predict[i].x4, xmax);
        ymin = min(predict[i].y1, ymin);
        ymin = min(predict[i].y2, ymin);
        ymin = min(predict[i].y3, ymin);
        ymin = min(predict[i].y4, ymin);
        ymax = max(predict[i].y1, ymax);
        ymax = max(predict[i].y2, ymax);
        ymax = max(predict[i].y3, ymax);
        ymax = max(predict[i].y4, ymax);
        cur_box.x1 = xmin;
        cur_box.x2 = xmax;
        cur_box.y1 = ymin;
        cur_box.y2 = ymax;
        
        cur_box.conf = predict[i].conf;
        float area = (xmax - xmin) * (ymax - ymin + 1.0);
        cur_box.area = area;
        BBox_4.push_back(cur_box);
    }
    return BBox_4;
}
float maximum(vector<OBBInfo4>predict,vector<pair<int,float>>scores,float QAQ,string property)
{
    float value = QAQ;
    for (int i = 1; i < scores.size(); i++)
    {
        value = max(predict[scores[i].first].get(property), value);
    }
    return value;
}
float minimum(vector<OBBInfo4> predict, vector<pair<int, float>> scores, float QAQ, string property)
{
    float value = QAQ;
    for (int i = 1; i < scores.size(); i++)
    {
        value = min(predict[scores[i].first].get(property), value);
    }
    return value;
}
//转换成Dou
vector<double> convert_vector(OBBInfo8 cur_box)
{
    vector<double> Demo;
    Demo.push_back(cur_box.x1);
    Demo.push_back(cur_box.y1);
    Demo.push_back(cur_box.x2);
    Demo.push_back(cur_box.y2);
    Demo.push_back(cur_box.x3);
    Demo.push_back(cur_box.y3);
    Demo.push_back(cur_box.x4);
    Demo.push_back(cur_box.y4);
    return Demo;
}
void non_max_supression(vector<OBBInfo8>&predict,float iou_thresh)
{
    vector<int> keep;//保留下来的碎银的操作
    vector<OBBInfo4> BBox_4 = Convert_8Points(predict);
    //保存下来的int数组就是原来的那个顺序
    for (int i = 0; i < BBox_4.size();i++)
    {
        vector<int> demo;
        for (int j = i + 1; j < BBox_4.size();)
        {
            float xx1 = max(BBox_4[i].x1,BBox_4[j].x1);
            float yy1 = max(BBox_4[i].y1, BBox_4[j].y1);
            float xx2 = min(BBox_4[i].x2, BBox_4[j].x2);
            float yy2 = max(BBox_4[i].y2, BBox_4[j].y2);
            float w = (std::max)(float(0), xx2 - xx1 + 1);
            float h = (std::max)(float(0), yy2 - yy1 + 1);
            float inter = w * h;
            float ovr = inter / (BBox_4[i].area + BBox_4[j].area - inter);
            double iou = iou_poly(convert_vector(predict[i]), convert_vector(predict[j]));
            if (iou >= iou_thresh&& ovr>iou_thresh)
            {
                BBox_4.erase(BBox_4.begin() + j);
                predict.erase(predict.begin() + j);
            }
            else
            {
                j++;
            }
        }
    }
}
int label_aqusision(vector<float> prob)
{
    float value = prob[0];
    int index=0;
    for (int i = 0; i < prob.size(); i++)
    {
        if(prob[i]>value)
        {
            index = i;
            value = prob[i];
        }
    }
    return index;
}
vector<OBBInfo8> non_max_supression_8_points(vector<OBBInfo8> predict,float conf_thresh,float iou_thresh)
{
    const int min_wh = 2;
    const int max_wh = 4096;
    const float alpha_1 = 0.55;
    const float alpha_2 = 0.45;
    //最大的检测数量
    const int max_det = 500;
    vector<OBBInfo8> BBox_filtered;
    //冗余框检测的算法
    for (int i = 0; i < predict.size();i++)
    {
        OBBInfo8 cur_box = predict[i];
        int index = label_aqusision(predict[i].prob);
        predict[i].conf = cur_box.conf * max_prob(cur_box.prob);
        predict[i].label = index;
        if (cur_box.conf > conf_thresh)
        {

            BBox_filtered.push_back(predict[i]);
        }
    }
    sort(BBox_filtered.begin(), BBox_filtered.end(), Conf_compare);
    //根据新的置信度再进行一步的筛选
    non_max_supression(BBox_filtered, iou_thresh);
    //仅仅支持单张推理的过程

    return BBox_filtered;
}