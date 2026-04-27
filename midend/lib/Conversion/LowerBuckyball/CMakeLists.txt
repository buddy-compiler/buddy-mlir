add_mlir_library(LowerBuckyballPass
  AssignBuckyballBanksPass.cpp
  LowerBuckyballToBankSSAPass.cpp
  ReportBankUsagePass.cpp
  LowerBuckyballPass.cpp

  LINK_LIBS PUBLIC
  BuddyBuckyballTransforms
  )
