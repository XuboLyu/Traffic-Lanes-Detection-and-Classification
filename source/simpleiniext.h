#ifndef __SIMPLE_INI_EXT_H__
#define __SIMPLE_INI_EXT_H__


#include "SimpleIni.h"

void split(std::string &s, std::string &delim, std::vector<std::string> &ret)
{
	size_t last = 0;
	size_t index = s.find_first_of(delim, last);
	while (index != std::string::npos) {
		ret.push_back(s.substr(last, index - last));
		last = index + 1;
		index = s.find_first_of(delim, last);
	}

	if (index - last > 0) {
		ret.push_back(s.substr(last, index - last));
	}
}

class CSimpleIniExt : public CSimpleIni
{
protected:
  bool silent_mode_;
public:

  CSimpleIniExt(bool silent_mode = false) : CSimpleIni()
  {
    silent_mode_ = silent_mode;
  }


  const bool GetValueExt(const char *section_name, const char *key_name, int &val) const
  {
    const char* rst = (char*)GetValue(section_name, key_name);
    if (rst) {
      int new_val = atoi(rst);
      if (!silent_mode_)
        printf("CSimpleIniExt: [%s] %s: %d -> %d\n", section_name, key_name, val, new_val);
      val = new_val;
      return true;
    } else {
      if (!silent_mode_)
        printf("CSimpleIniExt: [%s] %s: keep %d \n", section_name, key_name, val);
      return false;
    }
  }

  const bool GetValueExt(const char *section_name, const char *key_name, float &val) const
  {
    const char* rst = GetValue(section_name, key_name);
    if (rst) {
      float new_val = atof(rst);
      if (!silent_mode_)
        printf("CSimpleIniExt: [%s] %s: %f -> %f\n", section_name, key_name, val, new_val);
      val = new_val;
      return true;
    } else {
      if (!silent_mode_)
        printf("CSimpleIniExt: [%s] %s: keep %f \n", section_name, key_name, val);
      return false;
    }
  }

  const bool GetValueExt(const char *section_name, const char *key_name, std::string &val) const
  {
    const char* rst = GetValue(section_name, key_name);
    if (rst) {
      std::string new_val(rst);
      if (!silent_mode_)
        printf("CSimpleIniExt: [%s] %s: %s -> %s\n", section_name, key_name, val.c_str(), new_val.c_str());
      val = new_val;
      return true;
    } else {
      if (!silent_mode_)
        printf("CSimpleIniExt: [%s] %s: keep %s \n", section_name, key_name, val.c_str());
      return false;
    }
  }

  const bool GetValueExtList(const char *section_name, const char *key_name, std::vector<int> &val_v, const char *delim) const
  {
    const char* rst = (char*)GetValue(section_name, key_name);
    if (rst) {
      std::vector<std::string> items;
      std::string s_rst(rst), s_delim(delim);
      split(s_rst, s_delim, items);
      std::vector<int> new_val_v;
      for (uint i = 0; i < items.size(); i++) {
        if (items[i].length() > 0) {
          if (items[i][0] == '0' && items[i][1] == 'x') {
            int num = strtol(items[i].c_str(), NULL, 16);
            new_val_v.push_back(num);
          } else {
            new_val_v.push_back(atoi(items[i].c_str()));
          }
        }
      }
      if (!silent_mode_) {
        printf("CSimpleIniExt: [%s] %s: ", section_name, key_name);
        for (uint i = 0; i < val_v.size(); i++)
          printf("%d ", val_v[i]);
        printf("->");
        for (uint i = 0; i < new_val_v.size(); i++)
          printf("%d ", new_val_v[i]);
      }
      val_v = new_val_v;
      return true;
    } else {
      if (!silent_mode_) {
        printf("CSimpleIniExt: [%s] %s: keep \n", section_name, key_name);
        for (uint i = 0; i < val_v.size(); i++)
          printf("%d ", val_v[i]);
      }
      return false;
    }
  }



  const bool GetValueExtList(const char *section_name, const char *key_name, std::vector<float> &val_v, const char *delim) const
  {
    const char* rst = (char*)GetValue(section_name, key_name);
    if (rst) {
      std::vector<std::string> items;
      std::string s_rst(rst), s_delim(delim);
      split(s_rst, s_delim, items);
      std::vector<float> new_val_v;
      for (uint i = 0; i < items.size(); i++) {
        if (items[i].length() > 0) {
          new_val_v.push_back(atof(items[i].c_str()));
        }
      }
      if (!silent_mode_) {
        printf("CSimpleIniExt: [%s] %s: ", section_name, key_name);
        for (uint i = 0; i < val_v.size(); i++)
          printf("%f ", val_v[i]);
        printf("->");
        for (uint i = 0; i < new_val_v.size(); i++)
          printf("%f ", new_val_v[i]);
      }
      val_v = new_val_v;
      return true;
    } else {
      if (!silent_mode_) {
        printf("CSimpleIniExt: [%s] %s: keep \n", section_name, key_name);
        for (uint i = 0; i < val_v.size(); i++)
          printf("%f ", val_v[i]);
      }
      return false;
    }
  }


  const bool GetValueExtList(const char *section_name, const char *key_name, std::vector<std::string> &val_v, const char *delim) const
  {
    const char* rst = GetValue(section_name, key_name);
    if (rst) {
      std::vector<std::string> items;
      std::string s_rst(rst), s_delim(delim);
      split(s_rst, s_delim, items);
      std::vector<std::string> new_val_v;
      for (uint i = 0; i < items.size(); i++) {
        if (items[i].length() > 0) {
          new_val_v.push_back(items[i]);
        }
      }
      if (!silent_mode_) {
        printf("CSimpleIniExt: [%s] %s: ", section_name, key_name);
        for (uint i = 0; i < val_v.size(); i++)
          printf("%s ", val_v[i].c_str());
        printf("->");
        for (uint i = 0; i < new_val_v.size(); i++)
          printf("%s ", new_val_v[i].c_str());
      }
      val_v = new_val_v;
      return true;
    } else {
      if (!silent_mode_) {
        printf("CSimpleIniExt: [%s] %s: keep \n", section_name, key_name);
        for (uint i = 0; i < val_v.size(); i++)
          printf("%s ", val_v[i].c_str());
      }
      return false;
    }
  }

};

#endif