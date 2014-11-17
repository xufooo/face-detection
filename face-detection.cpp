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
# Last modified: 2014-11-17 14:51
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
Directory dir;
vector<string> filenames = dir.GetListFiles(".", "jpg", true);
for(int i = 0; i<filenames.size(); i++)
{
	Mat image=imread(filenames[i]);
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
}
}
}
//-- Show what you got
namedWindow( window_name, 1);
imshow( window_name, frame );
}
