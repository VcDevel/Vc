#!/usr/bin/env ruby

puts '#ifdef VC_GNU_ASM'

1.upto 8 do |max|
	print 'template<'
	max.downto 2 do |i|
		print "typename T#{i}, "
	end
	print "typename T1>\nstatic inline void ALWAYS_INLINE forceToRegisters("
	max.downto 2 do |i|
		print "const Vector<T#{i}> &x#{i}, "
	end
	print "const Vector<T1> &x1) {\n"
	print "  __asm__ __volatile__(\"\"::"
	max.downto 2 do |i|
		print "\"x\"(x#{i}.data()), "
	end
	print "\"x\"(x1.data()));\n}\n"

	print 'template<'
	max.downto 2 do |i|
		print "typename T#{i}, "
	end
	print "typename T1>\nstatic inline void ALWAYS_INLINE forceToRegistersDirty("
	max.downto 2 do |i|
		print "Vector<T#{i}> &x#{i}, "
	end
	print "Vector<T1> &x1) {\n"
	print "  __asm__ __volatile__(\"\":"
	max.downto 2 do |i|
		print "\"+x\"(x#{i}.data()), "
	end
	print "\"+x\"(x1.data()));\n}\n"
end

puts '#elif defined(VC_MSVC)'

1.upto 8 do |max|
	puts '#pragma optimize("g", off)'
	print 'template<'
	max.downto 2 do |i|
		print "typename T#{i}, "
	end
	print "typename T1>\nstatic inline void ALWAYS_INLINE forceToRegisters("
	max.downto 2 do |i|
		print "const Vector<T#{i}> &/*x#{i}*/, "
	end
	print "const Vector<T1> &/*x1*/) {\n"
	print "}\n"

	puts '#pragma optimize("g", off)'
	print 'template<'
	max.downto 2 do |i|
		print "typename T#{i}, "
	end
	print "typename T1>\nstatic inline void ALWAYS_INLINE forceToRegistersDirty("
	max.downto 2 do |i|
		print "Vector<T#{i}> &/*x#{i}*/, "
	end
	print "Vector<T1> &/*x1*/) {\n"
	print "}\n"
	puts '#pragma optimize("g", on)'
end

puts '#else'
puts '#error "forceToRegisters unsupported on this compiler"'
puts '#endif'
