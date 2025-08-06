import { ComponentFixture, TestBed } from '@angular/core/testing';

import { UploadLoaderComponent } from './upload-loader.component';

describe('LoaderComponent', () => {
  let component: UploadLoaderComponent;
  let fixture: ComponentFixture<UploadLoaderComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [UploadLoaderComponent]
    })
    .compileComponents();

    fixture = TestBed.createComponent(UploadLoaderComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
