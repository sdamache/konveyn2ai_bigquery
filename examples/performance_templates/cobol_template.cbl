      *================================================================
      * COPYBOOK: RECORD{ID}.CPY
      * DESCRIPTION: Performance test record structure {ID}
      * MODULE: {ID}
      *================================================================
       01  PERFORMANCE-RECORD-{ID}.
      *    HEADER SECTION
           05  REC-HEADER-{ID}.
               10  REC-ID              PIC 9(6) COMP.
               10  REC-TYPE            PIC X(4).
               10  REC-VERSION         PIC 9(3) COMP-3.
               10  REC-STATUS          PIC X(1).
               10  FILLER              PIC X(2).

      *    IDENTIFICATION SECTION
           05  REC-IDENTIFICATION-{ID}.
               10  CUSTOMER-ID         PIC 9(9) COMP.
               10  ACCOUNT-NUMBER      PIC X(16).
               10  SSN                 PIC 9(9) COMP.
               10  TAX-ID              PIC X(12).
               10  REFERENCE-NUM       PIC X(20).

      *    PERSONAL INFORMATION
           05  REC-PERSONAL-{ID}.
               10  LAST-NAME           PIC X(30).
               10  FIRST-NAME          PIC X(20).
               10  MIDDLE-INITIAL      PIC X(1).
               10  SUFFIX              PIC X(4).
               10  DATE-OF-BIRTH       PIC 9(8) COMP-3.
               10  GENDER-CODE         PIC X(1).
               10  MARITAL-STATUS      PIC X(1).

      *    ADDRESS INFORMATION
           05  REC-ADDRESS-{ID}.
               10  STREET-ADDRESS-1    PIC X(40).
               10  STREET-ADDRESS-2    PIC X(40).
               10  CITY                PIC X(30).
               10  STATE-CODE          PIC X(2).
               10  ZIP-CODE            PIC X(10).
               10  COUNTRY-CODE        PIC X(3).
               10  ADDRESS-TYPE        PIC X(1).

      *    FINANCIAL INFORMATION
           05  REC-FINANCIAL-{ID}.
               10  ANNUAL-INCOME       PIC S9(9)V99 COMP-3.
               10  CREDIT-SCORE        PIC 9(3) COMP-3.
               10  DEBT-TO-INCOME      PIC 9(3)V99 COMP-3.
               10  EMPLOYMENT-STATUS   PIC X(2).
               10  EMPLOYER-NAME       PIC X(40).
               10  JOB-TITLE           PIC X(30).
               10  YEARS-EMPLOYED      PIC 9(2) COMP-3.

      *    ACCOUNT DETAILS
           05  REC-ACCOUNT-{ID}.
               10  ACCOUNT-TYPE        PIC X(3).
               10  ACCOUNT-STATUS      PIC X(2).
               10  OPEN-DATE           PIC 9(8) COMP-3.
               10  LAST-ACTIVITY-DATE  PIC 9(8) COMP-3.
               10  CURRENT-BALANCE     PIC S9(11)V99 COMP-3.
               10  AVAILABLE-BALANCE   PIC S9(11)V99 COMP-3.
               10  CREDIT-LIMIT        PIC S9(9)V99 COMP-3.
               10  MINIMUM-PAYMENT     PIC S9(7)V99 COMP-3.

      *    SYSTEM FIELDS
           05  REC-SYSTEM-{ID}.
               10  CREATE-TIMESTAMP    PIC X(26).
               10  UPDATE-TIMESTAMP    PIC X(26).
               10  CREATE-USER-ID      PIC X(8).
               10  UPDATE-USER-ID      PIC X(8).
               10  RECORD-CHECKSUM     PIC X(32).
               10  FILLER              PIC X({FILLER_SIZE}).