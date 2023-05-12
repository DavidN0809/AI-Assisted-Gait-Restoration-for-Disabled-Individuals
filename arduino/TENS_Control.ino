#include <SPI.h>

byte ad1 = 0x00;
int CS1 = 10;
int CS2 = 9;
int CS3 = 6;
int CS4 = 5;
int i=0;
int data = 0;
int Quad, Hamstring, AntTib, Calf;

void setup(){
  pinMode (CS1, OUTPUT); // Calf
  pinMode (CS2, OUTPUT); // AntTib
  pinMode (CS3, OUTPUT); // Hamstring
  pinMode (CS4, OUTPUT); // Quad

  pinMode(13, OUTPUT);
  Serial.begin(9600);
  SPI.begin();
}

void loop() {
  if (Serial.available() > 0) {
    // Read the incoming data until a newline character is received
    String data = Serial.readStringUntil('\n');
    
    // Split the data into two strings separated by a comma
    int commaIndex1 = data.indexOf(',');
    int commaIndex2 = data.indexOf(',', commaIndex1 + 1);
    int commaIndex3 = data.indexOf(',', commaIndex2 + 1);

    String pair1String = data.substring(0, commaIndex1);
    String pair2String = data.substring(commaIndex1 + 1, commaIndex2);
    String pair3String = data.substring(commaIndex2 + 1, commaIndex3);
    String pair4String = data.substring(commaIndex3 + 1);
    
    // Convert the strings to integers and store them in arrays
    Quad = pair1String.toInt();
    Hamstring = pair2String.toInt();
    AntTib = pair3String.toInt();
    Calf = pair4String.toInt();
    
    // Clear the serial buffer
    while (Serial.available() > 0) 
      Serial.read();
  }

  digitalPotWrite(CS1, ad1, Calf);
  digitalPotWrite(CS2, ad1, AntTib);
  digitalPotWrite(CS3, ad1, Hamstring);
  digitalPotWrite(CS4, ad1, Quad);


  Serial.print("Received data: ");
  Serial.print("Quad: ");
    Serial.print(Quad);
    Serial.print(", Hamstring: ");
    Serial.print(Hamstring);
    Serial.print(", AntTib: ");
    Serial.print(AntTib);
    Serial.print(", Calf: ");
    Serial.println(Calf);
}

/*
void loop(){

  if (Serial.available() >= 2) {  // Wait until at least 2 bytes are available
    data = Serial.read() << 8 | Serial.read();  // Combine the two bytes into an integer
  }
  Serial.println(data);  // Print the received value

  digitalPotWrite(CS1, ad1, data);
  digitalPotWrite(CS2, ad1, data);
  digitalPotWrite(CS3, ad1, data);
  digitalPotWrite(CS4, ad1, data);

}
*/

int digitalPotWrite(int CS, byte address, int value){
  digitalWrite(CS, LOW);
  SPI.transfer(address);
  SPI.transfer(value);
  digitalWrite(CS, HIGH);
}