      * CUSTOMER RECORD COPYBOOK
      * Description: Customer master record layout
      * Version: 1.0
       01  CUSTOMER-RECORD.
           05  CUST-ID              PIC 9(8).
           05  CUST-NAME.
               10  CUST-LAST-NAME   PIC X(25).
               10  CUST-FIRST-NAME  PIC X(20).
               10  CUST-MIDDLE-INIT PIC X(1).
           05  CUST-ADDRESS.
               10  CUST-STREET      PIC X(30).
               10  CUST-CITY        PIC X(20).
               10  CUST-STATE       PIC X(2).
               10  CUST-ZIP         PIC 9(5).
               10  CUST-ZIP-EXT     PIC 9(4).
           05  CUST-PHONE           PIC 9(10).
           05  CUST-EMAIL           PIC X(50).
           05  CUST-STATUS          PIC X(1).
               88  CUST-ACTIVE      VALUE 'A'.
               88  CUST-INACTIVE    VALUE 'I'.
               88  CUST-SUSPENDED   VALUE 'S'.
           05  CUST-CREDIT-LIMIT    PIC 9(7)V99.
           05  CUST-BALANCE         PIC S9(7)V99.
           05  CUST-CREATED-DATE    PIC 9(8).
           05  CUST-UPDATED-DATE    PIC 9(8).
           05  FILLER               PIC X(10).