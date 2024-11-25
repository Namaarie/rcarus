use fs_extra::copy_items;
use fs_extra::dir::CopyOptions;
use std::env;

fn main() -> Result<(), ()> {
    // This tells cargo to rerun this script if something in res/ changes.
    // println!("cargo:rerun-if-changed=res/*");

    // if Ok("release".to_owned()) == env::var("PROFILE") {
    //     env::set_var("res", "./res/");
    //     Ok(())
    // } else {
    //     env::set_var("res", "./res/");
    // // Prepare what to copy and how
    // let mut copy_options = CopyOptions::new();
    // copy_options.overwrite = true;
    // let paths_to_copy = vec!["res/"];

    // // Copy the items to the directory where the executable will be built
    // let out_dir = env::var("OUT_DIR").unwrap();
    // copy_items(&paths_to_copy, out_dir, &copy_options).unwrap();

    // Copy the items to the directory where they will be hosted
    // - The out_dir will likely be different in your project
    // let out_dir = std::path::Path::new(&env::var("CARGO_MANIFEST_DIR")?)
    //     .parent().unwrap()
    //     .parent().unwrap()
    //     .parent().unwrap()
    //     .join("docs/.vuepress/public/res/tutorial9-models");
    // create_all(&out_dir, false)?;
    // copy_items(&paths_to_copy, out_dir, &copy_options)?;

    Ok(())
}
