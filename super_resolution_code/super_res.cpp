#include <iostream>

#include <visp/vpDebug.h>
#include <visp/vpImage.h>
#include <visp/vpImageIo.h>
#include <visp/vpDisplayX.h>


using namespace std ;


int main()
{
  vpImage<vpRGBa> I(480,319,0);
  vpImage<vpRGBa> Iimage(480,319);
  
  vpImageIo::read(Iimage,"../img/lion.jpg") ;
  
  vpDisplayX d1(Iimage) ;
  vpDisplay::display(Iimage) ;
  vpDisplay::flush(Iimage) ;
  vpDisplay::getClick(Iimage) ;


  return 0;
}
