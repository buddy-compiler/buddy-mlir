// RUN: buddy-opt %s --split-input-file --verify-diagnostics

// The former AME ISA operations are no longer registered. The AME dialect now
// implements ZTT v0.6 through ame.<mnemonic> and ame.intr.<mnemonic>.

module {
  // expected-error @+1 {{unregistered operation 'ame.msettilemi' found in dialect ('ame') that does not allow unknown operations}}
  "ame.msettilemi"() {tilem = 8 : i64} : () -> ()
}

// -----

module {
  // expected-error @+1 {{unregistered operation 'ame.mma.w.mm' found in dialect ('ame') that does not allow unknown operations}}
  "ame.mma.w.mm"() : () -> ()
}

// -----

module {
  // expected-error @+1 {{unregistered operation 'ame.intr.msettilemi' found in dialect ('ame') that does not allow unknown operations}}
  "ame.intr.msettilemi"() {tilem = 8 : i64} : () -> ()
}

// -----

module {
  // expected-error @+1 {{unregistered operation 'ame.intr.ztt.ame.release' found in dialect ('ame') that does not allow unknown operations}}
  "ame.intr.ztt.ame.release"() : () -> ()
}
