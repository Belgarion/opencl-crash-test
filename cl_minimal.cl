/**
 * Performs data transformation on 32 bit chunks (4 bytes) of data
 * using deterministic floating point operations on IEEE 754
 * compliant machines and devices.
 * @param *data     - pointer to in data (at least 32 bytes)
 * @param len       - length of data
 * @param index     - the current tile
 * @param *op       - pointer to the operator value
 * @param transform - flag indicates to transform the input data */
void cl_fp_operation(uchar *data, uint len, uint index,
                                  uint *op, uchar transform, uchar debug)
{
   uchar *temp;
   uint adjustedlen;
   int i, j, operand;
   float floatv, floatv1;
   float *floatp;
   
   /* Adjust the length to a multiple of 4 */
   adjustedlen = len & 0xfffffffc;

   /* Work on data 4 bytes at a time */
   for(i = 0; i < adjustedlen; i += 4)
   {
      /* Cast 4 byte piece to float pointer */
      if(transform) {
         floatp = (float *) &data[i];
      } else {
         floatv1 = *(float *) &data[i];
         floatp = &floatv1;
      }

      /* 4 byte separation order depends on initial byte:
       * #1) *op = data... determine floating point operation type
       * #2) operand = ... determine the value of the operand
       * #3) if(data[i ... determine the sign of the operand
       *                   ^must always be performed after #2) */
      switch(data[i] & 7)
      {
         case 0:
            *op += data[i + 1];
            operand = data[i + 2];
            if(data[i + 3] & 1) operand ^= 0x80000000;
            break;
         case 1:
            operand = data[i + 1];
            if(data[i + 2] & 1) operand ^= 0x80000000;
            *op += data[i + 3];
            break;
         case 2:
            *op += data[i];
            operand = data[i + 2];
            if(data[i + 3] & 1) operand ^= 0x80000000;
            break;
         case 3:
            *op += data[i];
            operand = data[i + 1];
            if(data[i + 2] & 1) operand ^= 0x80000000;
            break;
         case 4:
            operand = data[i];
            if(data[i + 1] & 1) operand ^= 0x80000000;
            *op += data[i + 3];
            break;
         case 5:
            operand = data[i];
            if(data[i + 1] & 1) operand ^= 0x80000000;
            *op += data[i + 2];
            break;
         case 6:
            *op += data[i + 1];
            operand = data[i + 1];
            if(data[i + 3] & 1) operand ^= 0x80000000;
            break;
         case 7:
            operand = data[i + 1];
            *op += data[i + 2];
            if(data[i + 3] & 1) operand ^= 0x80000000;
            break;
      } /* end switch(data[j] & 31... */

      /* Cast operand to float */
      floatv = operand;

      /* Replace pre-operation NaN with index */
      if(isnan(*floatp)) *floatp = index;

      /* Perform predetermined floating point operation */
      switch(*op & 3) {
         case 0:
            *floatp += floatv;
            break;
         case 1:
            *floatp -= floatv;
            break;
         case 2:
            *floatp *= floatv;
            break;
         case 3:
            *floatp /= floatv;
            break;
	 default:
	    break;
      }

      /* Replace post-operation NaN with index */
      if(isnan(*floatp)) *floatp = index;

      /* Add result of floating point operation to op */
      temp = (uchar *) floatp;
      for(j = 0; j < 4; j++) {
         *op += temp[j];
      }
   } /* end for(*op = 0... */
}

#define HASHLEN 32
__kernel void test() {
	uchar seed[4+HASHLEN];
	uint seedlen = 4+HASHLEN;
	for (int i = 0; i < seedlen; i++) {
		seed[i] = i;
	}
	uint index = 0;
	uint op;
	op = 1;
	uint transform = 0;
	cl_fp_operation(seed, seedlen, index, &op, transform, 0);
}
