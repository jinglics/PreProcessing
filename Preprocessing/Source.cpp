#include "preprocess.h"

int thresh = 100;
int max_thresh = 255;

Mat src, src_gray;
/// Function header
void thresh_callback(int, void*);

int main(){
  Mat image;
  int num=5;
  for (int i = 0; i < num; i++){
    cout << i << endl;
    image=imread("b"+to_string(i+1)+".JPG");
    Preprocess inst(image);
    inst.process();
    vector<Mat> candidates=inst.text_area_candidates(inst.get_result());
    int j = 0;
    for (auto it : candidates){
      //cout << "B" + to_string(i + 1) + "\\" + to_string(++j) + ".JPG"<<endl;
      imwrite("B" + to_string(i + 1) + "\\" + to_string(++j) + ".JPG", it);
    }
    candidates = inst.text_area_candidates(inst.get_invert());
    j = 0;
    for (auto it : candidates){
      imwrite("B" + to_string(i + 1) + "\\" + to_string(++j) + "in.JPG", it);
    }
  }
  //inst.text_area_candidates(inst.get_result());
  //inst.gaussion_blur(5);
  //imwrite("result.jpg", inst.get_result());
  //inst.rgb2gray();
  //imwrite("result.jpg", inst.get_result());
  //inst.erosion(0,4);
  //imwrite("result.jpg", inst.get_result());
  ////inst.binarization();
  //inst.dilation(0,4);
  //imwrite("result.jpg", inst.get_result());
  //inst.binarization();
  //imwrite("result.jpg", inst.get_result());
  //inst.gaussion_blur(5);

  //namedWindow("Test", CV_WINDOW_AUTOSIZE);
  //imshow("Test", inst.get_result());
  

  //src = inst.get_source();
  //src_gray = inst.get_result();
  //imwrite("result.jpg", src_gray);
   
  /// Create Window
  //char* source_window = "Source";
  //namedWindow(source_window, CV_WINDOW_AUTOSIZE);
  //imshow(source_window, src);

  //inst.eliminate_small_contour();


  waitKey(0);
}

/** @function thresh_callback */
void thresh_callback(int, void*)
{
  Mat threshold_output;
  vector<vector<Point> > contours;
  vector<Vec4i> hierarchy;

  /// Detect edges using Threshold
  //threshold(src_gray, threshold_output, thresh, 255, THRESH_BINARY);
  threshold(src_gray, threshold_output, thresh, 255, THRESH_BINARY);
  //Canny(src_gray, threshold_output, thresh, thresh * 2, 3);
  /// Find contours
  findContours(threshold_output, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
  
  /// Approximate contours to polygons + get bounding rects and circles
  vector<vector<Point> > contours_poly(contours.size());
  vector<Rect> boundRect(contours.size());


  for (int i = 0; i < contours.size(); i++)
  {
    approxPolyDP(Mat(contours[i]), contours_poly[i], 3, true);
    boundRect[i] = boundingRect(Mat(contours_poly[i]));
    //if(boundRect[i].area();
  }


  /// Draw polygonal contour + bonding rects + circles
  Mat drawing = Mat::zeros(threshold_output.size(), CV_8UC3);
  for (int i = 0; i< contours.size(); i++)
  {
    Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
    //drawContours(drawing, contours_poly, i, color, 1, 8, vector<Vec4i>(), 0, Point());
    rectangle(src, boundRect[i].tl(), boundRect[i].br(), color, 2, 8, 0);
    //circle(drawing, center[i], (int)radius[i], color, 2, 8, 0);
  }

  /// Show in a window
  namedWindow("Contours", CV_WINDOW_AUTOSIZE);
  imshow("Contours", src);
}

