CFLAGS = `pkg-config --cflags opencv`
LIBS = `pkg-config --libs opencv`

facedetect : face-detection.cpp
	g++ -g $(CFLAGS) $(LIBS) -Wl,-rpath=.:/usr/local/lib -o $@ $<
