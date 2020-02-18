string = ""

string = string.gsub(/\s+/, " ").split(/([^[:alnum:]])/)
string = string.keep_if do |c| c !~ /\s+/ && c != "" end
string = string.join("\n")

f = File.new("newfile", File::CREAT|File::TRUNC|File::RDWR, 0644)
f.write(string)
f.close
