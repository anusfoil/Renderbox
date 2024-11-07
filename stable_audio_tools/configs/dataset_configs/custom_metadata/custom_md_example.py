def get_custom_metadata(info):

    # Use relative path as the prompt
    relpath = info["relpath"]
    
    if "midi_score" in relpath:
        style_keword = ", plain synthesis"
    else:
        style_keword = ", expressive performance"
    
    prompt = ", ".join(relpath.split("/")[:-1]) + style_keword
    
    midi_path = "/".join(info['path'].split("/")[:-1]) + "/midi_score.mid"
    
    # print(prompt)
    return {"prompt": prompt, "midi": midi_path}