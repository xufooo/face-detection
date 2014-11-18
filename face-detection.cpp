/*=============================================================================
# Copyright (C) 2014 OoO
#
# This program is free software: you can redistribute it and/or modify 
# it under the terms of the GNU General Public License as published by 
# the Free Software Foundation, either version 3 of the License, or 
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful, 
# but WITHOUT ANY WARRANTY; without even the implied warranty of 
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the 
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License 
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
#
# Description: 
#
#
# Last modified: 2014-11-18 15:10
#
# Should you need to contact me, you can do so by 
# email - mail your message to <xufooo@gmail.com>.
=============================================================================*/
/**
* @file objectDetection2.cpp
* @author A. Huaman ( based in the classic facedetect.cpp in samples/c )
* @brief A simplified version of facedetect.cpp, show how to load a cascade classifier and how to find objects (Face + eyes) in a video stream - Using LBP here

ooo modiyied to adjust image detection
*/
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/contrib/contrib.hpp"
#include <iostream>
#include <stdio.h>
using namespace std;
using namespace cv;
/** Function Headers */
void detectAndDisplay( Mat frame );
Mat cropFace(Mat image, Point eye_1, Point eye_2, double offset_h, double offset_v, int dest_sz_w, int dest_sz_h);
/** Global variables */
String haar_dir = "./haarcascades/";
String lbp_dir = "./lbpcascades/";
String face_cascade_name = "haarcascade_frontalface_alt.xml";
String eyes_cascade_name = "haarcascade_eye_tree_eyeglasses.xml";
CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;
String window_name = "Image - Face detection";
/**
* @function main
*/
int main( void )
{
//VideoCapture capture;
const Mat *frame;
//-- 1. Load the cascade
if( !face_cascade.load( haar_dir + face_cascade_name ) ){ printf("--(!)Error loading face cascade\n"); return -1; };
if( !eyes_cascade.load( haar_dir + eyes_cascade_name ) ){ printf("--(!)Error loading eyes cascade\n"); return -1; };
//-- 2. Read images in the current folder 
//capture.open( -1 );
//if ( ! capture.isOpened() ) { printf("--(!)Error opening video capture\n"); return -1; }
//while ( capture.read(frame) )
//{
string input_path = "./samples/";
Directory dir;
vector<string> filenames = dir.GetListFiles(input_path , "jpg", true);
for(int i = 0; i<filenames.size(); i++)
{
	Mat image=imread(input_path + filenames[i]);
//	Mat image=imread("lena.jpg");
	frame=&image;
	if( frame->empty() )
	{
		printf(" --(!) No captured frame -- Break!");
		break;
	}
//-- 3. Apply the classifier to the frame
detectAndDisplay( *frame );
//-- bail out if escape was pressed
int c = waitKey();
if( (char)c == 27 ) { break; }
}
return 0;
}
/**
* @function detectAndDisplay
*/
void detectAndDisplay( Mat frame )
{
std::vector<Rect> faces;
Mat frame_gray;
cvtColor( frame, frame_gray, COLOR_BGR2GRAY );
equalizeHist( frame_gray, frame_gray );
//-- Detect faces
face_cascade.detectMultiScale( frame_gray, faces, 1.1, 2, 0, Size(80, 80) );
Point eye_xy[2];
for( size_t i = 0; i < faces.size(); i++ )
{
Mat faceROI = frame_gray( faces[i] );
std::vector<Rect> eyes;
//-- In each face, detect eyes
eyes_cascade.detectMultiScale( faceROI, eyes, 1.1, 2, 0 |CASCADE_SCALE_IMAGE, Size(30, 30) );
if( eyes.size() == 2)
{
//-- Draw the face
Point center( faces[i].x + faces[i].width/2, faces[i].y + faces[i].height/2 );
ellipse( frame, center, Size( faces[i].width/2, faces[i].height/2 ), 0, 0, 360, Scalar( 255, 0, 0 ), 2, 8, 0 );
for( size_t j = 0; j < eyes.size(); j++ )
{ //-- Draw the eyes
Point eye_center( faces[i].x + eyes[j].x + eyes[j].width/2, faces[i].y + eyes[j].y + eyes[j].height/2 );
int radius = cvRound( (eyes[j].width + eyes[j].height)*0.25 );
circle( frame, eye_center, radius, Scalar( 255, 0, 255 ), 3, 8, 0 );
eye_xy[j]=eye_center;
}
imwrite("./outputs/Test.jpg", cropFace(frame, eye_xy[0], eye_xy[1], 0.3, 0.3, 200, 200)); 
}
}
//-- Show what you got
imshow( window_name, frame );
}

Mat cropFace(Mat image, Point eye_1, Point eye_2, double offset_pct_h, double offset_pct_v, int dest_sz_w, int dest_sz_h)
{
	Point eye_left, eye_right;
	if(eye_1.x < eye_2.x)
		eye_left = eye_1, eye_right = eye_2;
	else
		eye_left = eye_2, eye_right = eye_1;

	cout<<"eye_left: "<<eye_left<<"\n"<<"eye_right: "<<eye_right<<"\n";
	Size dest_sz(dest_sz_w,dest_sz_h);
	int offset_h = cvFloor(offset_pct_h * dest_sz.width);	
	int offset_v = cvFloor(offset_pct_v * dest_sz.height);	
	cout<<"offset_h: "<<offset_h<<"\n"<<"offset_v: "<<offset_v<<"\n";
	double eye_direction_x = eye_right.x - eye_left.x;
	double eye_direction_y = eye_right.y - eye_left.y;
	cout<<"eye_direction_x: "<<eye_direction_x<<"\n"<<"eye_direction_y: "<<eye_direction_y<<"\n";
	double rotation = atan2(eye_direction_y, eye_direction_x);
	cout<<"rotation: "<<rotation<<"\n";
	double dist = sqrt(eye_direction_x * eye_direction_x + eye_direction_y * eye_direction_y);
	cout<<"dist: "<<dist<<"\n";
	double reference = dest_sz.width - 2.0 * offset_h;
	cout<<"reference: "<<reference<<"\n";
	double scale = dist / reference;
	cout<<"scale: "<<scale<<"\n";
	Mat rot_mat = getRotationMatrix2D(eye_left, rotation*180/3.14, 1);
	Mat image_r;
	warpAffine(image, image_r, rot_mat, image_r.size());
	imshow(window_name, image_r);
	waitKey();
	Rect crop_rect(eye_left.x - scale * offset_h, eye_left.y - scale * offset_v, dest_sz.width * scale, dest_sz.height * scale);
	cout<<"crop_rect: "<<crop_rect<<"\n";
	Mat cropped_image = image_r(crop_rect).clone();
	Mat resized_image;
	resize(cropped_image, resized_image, dest_sz);
	imshow(window_name, resized_image);
	waitKey();
	imshow(window_name, cropped_image);
	waitKey();
	return cropped_image;
}
