# VistA MUMPS Data Dictionary Export
# Performance Test Module {ID}
# File: PERF_TEST_{ID}

^DD({FILE_NUMBER},0)="PERFORMANCE TEST FILE {ID}^NL^1^1"
^DD({FILE_NUMBER},0,"NM","PERFORMANCE TEST FILE {ID}")=""

# Field .01: NAME
^DD({FILE_NUMBER},.01,0)="NAME^RF^^0;1^K:$L(X)>30!(X?.N)!($L(X)<3) X"
^DD({FILE_NUMBER},.01,1,0)="^.1"
^DD({FILE_NUMBER},.01,1,1,0)="{FILE_NUMBER}^B"
^DD({FILE_NUMBER},.01,3)="Enter the name of the test record, 3-30 characters"

# Field 1: TEST ID
^DD({FILE_NUMBER},1,0)="TEST ID^RF^^0;2^K:+X'=X!(X>999999)!(X<1) X"
^DD({FILE_NUMBER},1,3)="Type a Number between 1 and 999999"

# Field 2: DESCRIPTION
^DD({FILE_NUMBER},2,0)="DESCRIPTION^F^^0;3^K:$L(X)>80!($L(X)<1) X"
^DD({FILE_NUMBER},2,3)="Enter a description for this test record, 1-80 characters"

# Field 3: STATUS
^DD({FILE_NUMBER},3,0)="STATUS^S^A:ACTIVE;I:INACTIVE;P:PENDING;C:COMPLETED;^0;4^Q"
^DD({FILE_NUMBER},3,3)="Enter the current status of this test record"

# Field 4: SCORE
^DD({FILE_NUMBER},4,0)="SCORE^N^^0;5^K:+X'=X!(X>100)!(X<0) X"
^DD({FILE_NUMBER},4,3)="Type a Number between 0 and 100"

# Global Storage Definition
^DD({FILE_NUMBER},"GL",0,1)="^PERF_TEST_{ID}("

# Sample Data Records
^PERF_TEST_{ID}(1)="Performance Test Record {ID} Alpha^{TEST_ID_1}^Alpha test record for module {ID}^A^{SCORE_1}"
^PERF_TEST_{ID}(2)="Performance Test Record {ID} Beta^{TEST_ID_2}^Beta test record for module {ID}^P^{SCORE_2}"
^PERF_TEST_{ID}(3)="Performance Test Record {ID} Gamma^{TEST_ID_3}^Gamma test record for module {ID}^C^{SCORE_3}"

# Index Definitions
^DD({FILE_NUMBER},"IX",0)="^.11"
^DD({FILE_NUMBER},"IX",.01)=""
^DD({FILE_NUMBER},"IX","B")=""